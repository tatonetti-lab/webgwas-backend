use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use itertools::izip;
use polars::prelude::*;
use polars::series::arithmetic::LhsNumOps;

use crate::models::{Constant, Feature, Node, NodeType, Operators, ParsingNode};

pub fn parse_string_definition(phenotype_definition: &str) -> Result<Vec<ParsingNode>> {
    let mut nodes = Vec::new();
    let mut current_node: ParsingNode;
    for token in phenotype_definition.split_whitespace() {
        match token.chars().next() {
            Some('"') => {
                if !token.ends_with('"') {
                    return Err(anyhow::anyhow!("Invalid field name {}", token));
                }
                current_node =
                    ParsingNode::Feature(token.get(1..token.len() - 1).unwrap().to_string());
                nodes.push(current_node);
            }
            Some('`') => {
                if !token.ends_with('`') {
                    return Err(anyhow::anyhow!("Invalid operator {}", token));
                }
                let operator = Operators::from_str(token.get(1..token.len() - 1).unwrap())
                    .context(anyhow!("Unknown operator {}", token))?;
                current_node = ParsingNode::Operator(operator);
                nodes.push(current_node);
            }
            Some('<') => {
                if !token.ends_with('>') {
                    return Err(anyhow::anyhow!("Invalid constant {}", token));
                }
                current_node = ParsingNode::Constant(
                    token
                        .get(1..token.len() - 1)
                        .unwrap()
                        .parse::<f32>()
                        .unwrap(),
                );
                nodes.push(current_node);
            }
            _ => {
                return Err(anyhow::anyhow!("Invalid token {}", token));
            }
        }
    }
    Ok(nodes)
}

#[derive(Clone, Default)]
pub struct KnowledgeBase {
    cohort_id_code_to_field: HashMap<(i32, String), Feature>,
}

impl KnowledgeBase {
    pub fn new(fields: Vec<Feature>) -> Self {
        let cohort_id_code_to_field = fields
            .clone()
            .into_iter()
            .map(|f| ((f.cohort_id, f.code.clone()), f))
            .collect();
        Self {
            cohort_id_code_to_field,
        }
    }

    pub fn find_field(&self, cohort_id: i32, code: &str) -> Option<&Feature> {
        self.cohort_id_code_to_field
            .get(&(cohort_id, code.to_string()))
    }
}

pub fn validate_nodes(
    cohort_id: i32,
    nodes: &[ParsingNode],
    kb: &KnowledgeBase,
) -> Result<Vec<Node>> {
    let mut result = Vec::new();
    for node in nodes {
        match node {
            ParsingNode::Feature(field_code) => {
                let field = kb
                    .find_field(cohort_id, field_code)
                    .ok_or(anyhow::anyhow!("Unknown field {}", field_code))?
                    .clone();
                result.push(Node::Feature(field));
            }
            ParsingNode::Operator(_) => {
                result.push((*node).clone().into());
            }
            ParsingNode::Constant(value) => {
                let constant = Node::Constant(Constant {
                    value: *value,
                    node_type: NodeType::Real,
                });
                result.push(constant);
            }
        }
    }
    Ok(result)
}

pub fn type_check_nodes(nodes: &[Node]) -> Result<()> {
    let mut stack = Vec::new();
    for node in nodes {
        match node {
            Node::Feature(_) | Node::Constant(_) => {
                stack.push(node.clone());
            }
            Node::Operator(op) => {
                let operator_value = op.value();
                for _ in 0..operator_value.arity {
                    let top = stack.pop().ok_or(anyhow::anyhow!(
                        "Operator {} expects {} arguments, got {}",
                        operator_value.name,
                        operator_value.arity,
                        stack.len()
                    ))?;
                    match top {
                        Node::Operator(_) => {
                            bail!(
                                "Operator {} expects {} arguments, got {}",
                                operator_value.name,
                                operator_value.arity,
                                stack.len()
                            )
                        }
                        Node::Feature(field) => {
                            if field.node_type != operator_value.input_type
                                && operator_value.input_type != NodeType::Any
                            {
                                bail!(
                                    "Type mismatch: expected {}, got {}",
                                    operator_value.input_type,
                                    field.node_type
                                );
                            }
                        }
                        Node::Constant(constant) => {
                            if constant.node_type != operator_value.input_type
                                && operator_value.input_type != NodeType::Any
                            {
                                bail!(
                                    "Type mismatch: expected {}, got {}",
                                    operator_value.input_type,
                                    constant.node_type
                                );
                            }
                        }
                    };
                }
                let result_node = Node::Constant(Constant {
                    value: 0.0,
                    node_type: operator_value.output_type,
                });
                stack.push(result_node);
            }
        };
    }
    if stack.len() != 1 {
        bail!("Invalid definition stack: {:?}", stack);
    }
    Ok(())
}

pub fn validate_phenotype_definition(
    cohort_id: i32,
    definition: &str,
    kb: &KnowledgeBase,
) -> Result<Vec<Node>> {
    let nodes = parse_string_definition(definition)?;
    let valid_nodes = validate_nodes(cohort_id, &nodes, kb)?;
    if let Err(err) = type_check_nodes(&valid_nodes) {
        bail!("Phenotype definition is invalid: {}", err);
    }
    Ok(valid_nodes)
}

pub fn apply_phenotype_definition(definition: &[Node], df: &DataFrame) -> Result<Series> {
    let mut stack: Vec<Series> = Vec::new();
    for node in definition {
        match node {
            Node::Feature(field) => {
                stack.push(
                    df.column(field.code.as_str())
                        .unwrap()
                        .as_series()
                        .unwrap()
                        .clone(),
                );
            }
            Node::Operator(op) => {
                let operator_value = op.value();
                match operator_value.arity {
                    1 => {
                        let item = stack.pop().ok_or(anyhow::anyhow!(
                            "Operator {} expects {} arguments, got {}",
                            operator_value.name,
                            operator_value.arity,
                            stack.len()
                        ))?;
                        match op {
                            Operators::Root => {
                                stack.push(item.clone());
                            }
                            Operators::Not => {
                                let result = 1_f32.sub(&item);
                                stack.push(result);
                            }
                            _ => {
                                bail!("Unknown operator {} with arity 1", operator_value.name)
                            }
                        }
                    }
                    2 => {
                        let item2 = stack.pop().ok_or(anyhow::anyhow!(
                            "Operator {} expects {} arguments, got {}",
                            operator_value.name,
                            operator_value.arity,
                            stack.len()
                        ))?;
                        let item1 = stack.pop().ok_or(anyhow::anyhow!(
                            "Operator {} expects {} arguments, got {}",
                            operator_value.name,
                            operator_value.arity,
                            stack.len()
                        ))?;
                        match op {
                            Operators::Add => {
                                let result = item1 + item2;
                                stack.push(result?);
                            }
                            Operators::Sub => {
                                let result = item1 - item2;
                                stack.push(result?);
                            }
                            Operators::Mul => {
                                let result = item1 * item2;
                                stack.push(result?);
                            }
                            Operators::Div => {
                                let result = item1 / item2;
                                stack.push(result?);
                            }
                            Operators::And => {
                                let result = izip!(
                                    item1
                                        .f32()?
                                        .into_iter()
                                        .map(|x| x.expect("Item1 not found")),
                                    item2
                                        .f32()?
                                        .into_iter()
                                        .map(|x| x.expect("Item2 not found"))
                                )
                                .map(|(x, y)| x.min(y))
                                .collect();
                                stack.push(result);
                            }
                            Operators::Or => {
                                let result = izip!(
                                    item1
                                        .f32()?
                                        .into_iter()
                                        .map(|x| x.expect("Item1 not found")),
                                    item2
                                        .f32()?
                                        .into_iter()
                                        .map(|x| x.expect("Item2 not found"))
                                )
                                .map(|(x, y)| x.max(y))
                                .collect();
                                stack.push(result);
                            }
                            Operators::Gt => {
                                let result = item1.gt(&item2)?.into_series();
                                stack.push(result);
                            }
                            Operators::Ge => {
                                let result = item1.gt_eq(&item2)?.into_series();
                                stack.push(result);
                            }
                            Operators::Lt => {
                                let result = item1.lt(&item2)?.into_series();
                                stack.push(result);
                            }
                            Operators::Le => {
                                let result = item1.lt_eq(&item2)?.into_series();
                                stack.push(result);
                            }
                            Operators::Eq => {
                                let result = item1.equal(&item2)?.into_series();
                                stack.push(result);
                            }
                            _ => {
                                bail!("Unknown operator {} with arity 2", operator_value.name)
                            }
                        }
                    }
                    _ => {
                        bail!(
                            "Unknown operator {} with arity {}",
                            operator_value.name,
                            operator_value.arity
                        )
                    }
                };
            }
            Node::Constant(constant) => {
                let constant_value = constant.value;
                let result = std::iter::repeat(constant_value)
                    .take(df.height())
                    .collect();
                stack.push(result);
            }
        };
    }
    if stack.len() != 1 {
        bail!("Invalid definition stack: {:?}", stack);
    }
    Ok(stack[0].clone())
}
