use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{anyhow, bail, Context, Result};
use faer::Mat;

use crate::models::{Constant, Feature, Node, NodeType, Operators, ParsingNode};

pub fn parse_string_definition(phenotype_definition: &str) -> Result<Vec<ParsingNode>> {
    let mut nodes = Vec::new();
    let mut current_node: ParsingNode;
    for token in phenotype_definition.split_whitespace() {
        match token.chars().next() {
            Some('"') => {
                if !token.ends_with('"') {
                    bail!("Invalid field name {}", token);
                }
                current_node =
                    ParsingNode::Feature(token.get(1..token.len() - 1).unwrap().to_string());
                nodes.push(current_node);
            }
            Some('`') => {
                if !token.ends_with('`') {
                    bail!("Invalid operator {}", token);
                }
                let operator = Operators::from_str(token.get(1..token.len() - 1).unwrap())
                    .context(anyhow!("Unknown operator {}", token))?;
                current_node = ParsingNode::Operator(operator);
                nodes.push(current_node);
            }
            Some('<') => {
                if !token.ends_with('>') {
                    bail!("Invalid constant {}", token);
                }
                current_node = ParsingNode::Constant(Constant::from_str(token)?);
                nodes.push(current_node);
            }
            _ => {
                bail!("Invalid token {}", token);
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
            ParsingNode::Constant(inner) => {
                let constant = Node::Constant(inner.clone());
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
    let valid_nodes = validate_nodes(cohort_id, &nodes, kb).context("Error validating nodes")?;
    type_check_nodes(&valid_nodes).context("Error type checking nodes")?;
    Ok(valid_nodes)
}

pub fn apply_phenotype_definition(
    definition: &[Node],
    names: &[String],
    phenotypes: &Mat<f32>,
) -> Result<Vec<f32>> {
    let mut stack = Vec::new();
    for node in definition {
        match node {
            Node::Feature(field) => {
                let idx = names.iter().position(|x| *x == field.code).unwrap();
                let column = phenotypes.col(idx).iter().copied().collect::<Vec<f32>>();
                stack.push(column);
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
                                stack.push(item);
                            }
                            Operators::Not => {
                                let result =
                                    item.iter().map(|x| (1.0_f32 - x)).collect::<Vec<f32>>();
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
                                let result =
                                    item1.iter().zip(item2.iter()).map(|(x, y)| x + y).collect();
                                stack.push(result);
                            }
                            Operators::Sub => {
                                let result =
                                    item1.iter().zip(item2.iter()).map(|(x, y)| x - y).collect();
                                stack.push(result);
                            }
                            Operators::Mul => {
                                let result =
                                    item1.iter().zip(item2.iter()).map(|(x, y)| x * y).collect();
                                stack.push(result);
                            }
                            Operators::Div => {
                                let result =
                                    item1.iter().zip(item2.iter()).map(|(x, y)| x / y).collect();
                                stack.push(result);
                            }
                            Operators::And => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| x.min(*y))
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Or => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| x.max(*y))
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Gt => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Ge => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| if x >= y { 1.0 } else { 0.0 })
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Lt => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Le => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| if x <= y { 1.0 } else { 0.0 })
                                    .collect();
                                stack.push(result);
                            }
                            Operators::Eq => {
                                let result = item1
                                    .iter()
                                    .zip(item2.iter())
                                    .map(|(x, y)| if x == y { 1.0 } else { 0.0 })
                                    .collect();
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
                    .take(phenotypes.nrows())
                    .collect();
                stack.push(result);
            }
        };
    }
    if stack.len() != 1 {
        bail!("Invalid definition stack: {:?}", stack);
    }
    let result = stack.pop().unwrap();
    Ok(result)
}

/// Convert a phenotype definition (reverse polish notation nodes) to a string
/// e.g. ["age", 30, "gt" "sex" "male" "eq" "and"] -> "AND(GT('age', 30), EQ('sex', 'male'))"
pub fn format_phenotype_definition(nodes: &[Node]) -> String {
    let mut stack: Vec<String> = Vec::new();
    for node in nodes {
        match node {
            Node::Feature(_) => {
                let formatted_node = format_node(node);
                stack.push(formatted_node);
            }
            Node::Constant(_) => {
                let formatted_node = format_node(node);
                stack.push(formatted_node);
            }
            Node::Operator(op) => {
                let operator_value = op.value();
                let operator_string = format_node(node);
                let mut this_string = format!("{}(", operator_string);
                assert!(operator_value.arity > 0);
                let mut local_stack = Vec::new();
                for _ in 0..operator_value.arity {
                    let top = stack.pop().unwrap();
                    local_stack.push(top);
                }
                for top in local_stack.iter().rev() {
                    this_string.push_str(format!("{}, ", top).as_str());
                }
                // Remove the last two characters = ", "
                this_string.pop();
                this_string.pop();
                this_string.push(')');
                stack.push(this_string);
            }
        };
    }
    stack.pop().expect("Stack is empty")
}

pub fn format_node(node: &Node) -> String {
    match node {
        Node::Feature(field) => format!("'{}' [{}]", field.name, field.code),
        Node::Operator(op) => op.to_string(),
        Node::Constant(constant) => format!("`{}`", constant.value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_phenotype_definition() {
        let nodes = vec![
            Node::Feature(Feature {
                id: 0,
                code: "1".to_string(),
                name: "age".to_string(),
                node_type: NodeType::Real,
                sample_size: 0,
                cohort_id: 0,
            }),
            Node::Constant(Constant {
                value: 30.0,
                node_type: NodeType::Real,
            }),
            Node::Operator(Operators::Gt),
            Node::Feature(Feature {
                id: 0,
                code: "2".to_string(),
                name: "sex".to_string(),
                node_type: NodeType::Bool,
                sample_size: 0,
                cohort_id: 0,
            }),
            Node::Feature(Feature {
                id: 0,
                code: "3".to_string(),
                name: "male".to_string(),
                node_type: NodeType::Bool,
                sample_size: 0,
                cohort_id: 0,
            }),
            Node::Operator(Operators::Eq),
            Node::Operator(Operators::And),
        ];
        let result = format_phenotype_definition(&nodes);
        assert_eq!(
            result,
            "AND(GT('age' [1], `30`), EQ('sex' [2], 'male' [3]))"
        );
    }
}
