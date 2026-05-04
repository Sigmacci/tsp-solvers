use std::collections::VecDeque;
use std::any::Any;
use std::collections::HashMap;
use std::collections::HashSet;
use std::{env, vec};
use std::error::Error;
use std::fs;
use std::io::Write;
use rand::Rng;
use rand::seq::{IteratorRandom, SliceRandom};
use std::time;
use std::fmt;
use std::cmp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Move {
    ExtInsert { node: usize, after_idx: usize },
    ExtRemove { idx: usize },
    IntVertexSwap { idx1: usize, idx2: usize },
    IntEdgeSwap { reverse_start: usize, reverse_end: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NeighborhoodType {
    VertexSwap,
    EdgeSwap
}

impl fmt::Display for NeighborhoodType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NeighborhoodType::VertexSwap => write!(f, "Vertex Swap"),
            NeighborhoodType::EdgeSwap => write!(f, "Edge Swap"),
        }
    }
}



fn read_csv_mapped(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>>{
    let     content : String        = fs::read_to_string(file_path)?;
    let mut result  : Vec<Vec<f64>> = Vec::new();

    for (_, line) in content.lines().enumerate() {
        if line.trim().is_empty() { continue }

        let row : Result<Vec<f64>, _> = line.split(';')
                                            .map(|value| value.trim().parse::<f64>())
                                            .collect();
        result.push(row?);
    }

    Ok(result)
}

fn get_distance_matrix_and_rewards(coordinates: Vec<Vec<f64>>) -> (Vec<Vec<i64>>, Vec<i64>) {
    let     num_points      : usize         = coordinates.len();
    let mut distance_matrix : Vec<Vec<i64>> = vec![vec![0; num_points]; num_points];
    let mut rewards         : Vec<i64>      = vec![0; num_points];

    for i in 0..num_points {
        for j in 0..num_points {
            if i != j {
                let dx = coordinates[i][0] - coordinates[j][0];
                let dy = coordinates[i][1] - coordinates[j][1];

                distance_matrix[i][j] = ((dx * dx + dy * dy).sqrt() + 0.5).floor() as i64;
                distance_matrix[j][i] = distance_matrix[i][j];
            }
        }
        rewards[i] = coordinates[i][2] as i64;
    }
    (distance_matrix, rewards)
}

fn solve_random(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut total_score   : i64      = 0;
    let mut total_length  : i64      = 0;

    for i in 0.._visit_subset.len() {
        let from = _visit_subset[i       % _visit_subset.len()] as usize;
        let to   = _visit_subset[(i + 1) % _visit_subset.len()] as usize;

        total_score += rewards[from] - distance_matrix[from][to];
        total_length += distance_matrix[from][to];
    }

    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut _visit_subset, total_length, total_score);
        return (route, total_score, total_length);
    }
    (_visit_subset, total_score, total_length)
}

fn solve_2_regret(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    solve_2_regret_weighted(distance_matrix, rewards, visit_subset, -1.0, 1.0, phase2)
}

fn solve_2_regret_weighted(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, weight_best: f64, weight_second: f64, phase2: bool) -> (Vec<usize>, i64, i64) {
    if visit_subset.len() < 2 {
        return (Vec::new(), 0, 0);
    }
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut total_score   : i64        = 0;
    let mut total_length  : i64        = 0;
    let mut route         : Vec<usize> = Vec::new();

    route.push(_visit_subset.swap_remove(0));

    let mut best_point : usize = 0;
    let mut best_score : i64   = i64::MIN;
    for i in 0.._visit_subset.len() {
        let score = rewards[_visit_subset[i]] - distance_matrix[route[0]][_visit_subset[i]];
        if score > best_score {
            best_score = score;
            best_point = i;
        }
    }

    route.push(_visit_subset.swap_remove(best_point));
    total_score += rewards[route[0]] - distance_matrix[route[0]][route[1]] + rewards[route[1]] - distance_matrix[route[1]][route[0]];

    while !_visit_subset.is_empty() {
        let mut best_regret  : i64   = i64::MIN;
        let mut best_point   : usize = 0;
        let mut best_cost    : i64   = 0;
        let mut best_postion : usize = 0;
        let mut best_idx     : usize = 0;

        for (idx, &point) in _visit_subset.iter().enumerate() {
            let mut best_insertion_cost    : i64   = i64::MAX;
            let mut second_best_cost       : i64   = i64::MAX;
            let mut insertion_index        : usize = 0;

            for i in 0..route.len() {
                let from : usize = route[i       % route.len()];
                let to   : usize = route[(i + 1) % route.len()];

                let insertion_cost = distance_matrix[from][point] + distance_matrix[point][to] - distance_matrix[from][to];

                if insertion_cost < best_insertion_cost {
                    second_best_cost       = best_insertion_cost;
                    best_insertion_cost    = insertion_cost;
                    insertion_index        = i + 1;
                } else if insertion_cost < second_best_cost {
                    second_best_cost       = insertion_cost;
                }
            }

            let regret = (weight_second * (second_best_cost as f64) + weight_best * (best_insertion_cost as f64) + 0.5).floor() as i64;
            if regret > best_regret {
                best_regret  = regret;
                best_point   = point;
                best_cost    = best_insertion_cost;
                best_postion = insertion_index;
                best_idx     = idx;
            }
        }
        route.insert(best_postion, best_point);
        total_score += rewards[best_point] - best_cost;
        total_length += best_cost;
        _visit_subset.swap_remove(best_idx);
    }

    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut route, total_length, total_score);
        return (route, total_score, total_length);
    }
    (route, total_score, total_length)
}

fn solve_2_regret_weighted_wrapper(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    solve_2_regret_weighted(distance_matrix, rewards, visit_subset, -1.5, 2.0, phase2)
}

fn solve_greedy_nn(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut route         : Vec<usize> = Vec::new();
    let mut total_score   : i64        = 0;
    let mut total_length  : i64        = 0;

    route.push(_visit_subset.remove(0));
    total_score += rewards[route[0]];
    while !_visit_subset.is_empty() {
        let last_point = *route.last().unwrap();
        let mut best_point : usize = 0;
        let mut best_score : i64   = i64::MIN;

        for &point in &_visit_subset {
            let score = rewards[point] - distance_matrix[last_point][point];
            if score > best_score {
                best_score = score;
                best_point = point;
            }
        }
        route.push(best_point);
        total_score += best_score;
        total_length += distance_matrix[last_point][best_point];
        _visit_subset.retain(|&x| x != best_point);
    }
    total_score += -distance_matrix[*route.last().unwrap()][route[0]];
    total_length += distance_matrix[*route.last().unwrap()][route[0]];
    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut route, total_length, total_score);
        return (route, total_score, total_length);
    }
    (route, total_score, total_length)
}

fn solve_greedy_nna(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut route         : Vec<usize> = Vec::new();
    let mut total_score   : i64        = 0;
    let mut total_length  : i64        = 0;

    route.push(_visit_subset.remove(0));
    total_score += rewards[route[0]];
    while !_visit_subset.is_empty() {
        let last_point = *route.last().unwrap();
        let mut best_point : usize = 0;
        let mut best_score : i64 = i64::MIN;

        for &point in &_visit_subset {
            let score = -distance_matrix[last_point][point];
            if score > best_score {
                best_score = score;
                best_point = point;
            }
        }
        route.push(best_point);
        _visit_subset.retain(|&x| x != best_point);
        total_score += rewards[best_point] + best_score;
        total_length += distance_matrix[last_point][best_point];
    }
    total_score += -distance_matrix[*route.last().unwrap()][route[0]];
    total_length += distance_matrix[*route.last().unwrap()][route[0]];
    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut route, total_length, total_score);
        return (route, total_score, total_length);
    }

    (route, total_score, total_length)
}

fn solve_greedy_gc(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut route         : Vec<usize> = Vec::new();
    let mut total_score   : i64        = 0;
    let mut total_length  : i64        = 0;

    route.push(_visit_subset.remove(0));
    total_score += rewards[route[0]];

    let mut best_point : usize = 0;
    let mut best_score : i64 = i64::MIN;
    let last_point = *route.last().unwrap();
    for &point in &_visit_subset {
        let score = rewards[point] - distance_matrix[last_point][point];
        if score > best_score {
            best_score = score;
            best_point = point;
        }
    }

    route.push(best_point);
    total_score += best_score - distance_matrix[last_point][best_point];
    total_length += distance_matrix[last_point][best_point];
    _visit_subset.retain(|&x| x != best_point);

    while !_visit_subset.is_empty() {
        let mut best_point = 0;
        let mut best_pos = 0;
        let mut best_delta = i64::MIN;
        for &point in &_visit_subset {
            for i in 0..route.len() {
                let prev = route[i];
                let next = route[(i + 1) % route.len()];
                let delta =
                    rewards[point]
                    - distance_matrix[prev][point]
                    - distance_matrix[point][next]
                    + distance_matrix[prev][next];
                if delta > best_delta {
                    best_delta = delta;
                    best_point = point;
                    best_pos = i + 1;
                }
            }
        }
        route.insert(best_pos, best_point);
        total_score += best_delta;
        total_length += -best_delta + rewards[best_point];
        _visit_subset.retain(|&x| x != best_point);
    }

    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut route, total_length, total_score);
        return (route, total_score, total_length);
    }
    (route, total_score, total_length)
}

fn solve_greedy_gca(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<usize>, phase2: bool) -> (Vec<usize>, i64, i64) {
    let mut _visit_subset : Vec<usize> = visit_subset.clone();
    let mut route         : Vec<usize> = Vec::new();
    let mut total_score   : i64        = 0;
    let mut total_length  : i64        = 0;

    route.push(_visit_subset.remove(0));
    total_score += rewards[route[0]];
    let mut best_point : usize = 0;
    let mut best_score : i64 = i64::MIN;
    let last_point = *route.last().unwrap();
    for &point in &_visit_subset {
        let score = -distance_matrix[last_point][point];
        if score > best_score {
            best_score = score;
            best_point = point;
        }
    }

    route.push(best_point);
    total_score += rewards[best_point] + best_score - distance_matrix[last_point][best_point];
    total_length += distance_matrix[last_point][best_point] - best_score;
    _visit_subset.retain(|&x| x != best_point);

    while !_visit_subset.is_empty() {
        let mut best_point = 0;
        let mut best_pos = 0;
        let mut best_delta = i64::MIN;
        for &point in &_visit_subset {
            for i in 0..route.len() {
                let prev = route[i];
                let next = route[(i + 1) % route.len()];
                let delta =
                    - distance_matrix[prev][point]
                    - distance_matrix[point][next]
                    + distance_matrix[prev][next];
                if delta > best_delta {
                    best_delta = delta;
                    best_point = point;
                    best_pos = i + 1;
                }
            }
        }
        route.insert(best_pos, best_point);
        total_score += rewards[best_point] + best_delta;
        _visit_subset.retain(|&x| x != best_point);
    }

    if phase2 {
        let (route, total_score, total_length) = solve_greedy_phase2(distance_matrix, rewards, &mut route, total_length, total_score);
        return (route, total_score, total_length);
    }
    (route, total_score, total_length)
}

fn solve_greedy_phase2(distance_matrix: &[Vec<i64>], rewards: &[i64], route: &mut Vec<usize>, total_length: i64, total_score: i64) -> (Vec<usize>, i64, i64) {
    let mut current_score = total_score;
    let mut current_length = total_length;

    let mut improved = true;

    while improved && route.len() > 2 { 
        improved = false;
        
        for i in 0..route.len() {
            let prev = route[(i + route.len() - 1) % route.len()];
            let current = route[i];
            let next = route[(i + 1) % route.len()];
            
            let cost_of_removal = distance_matrix[prev][next] 
                                - (distance_matrix[prev][current] + distance_matrix[current][next] - rewards[current]);
            
            if cost_of_removal < 0 {
                current_score -= cost_of_removal;
                
                current_length += distance_matrix[prev][next] 
                                - distance_matrix[prev][current] 
                                - distance_matrix[current][next];

                route.remove(i);
                improved = true;
                break; 
            }
        }
    }

    (route.clone(), current_score, current_length)
}

fn calculate_score(route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> i64 {
    let mut total_score: i64 = 0;
    
    for i in 0..(route.len() - 1) {
        let from = route[i    ];
        let to   = route[i + 1];

        total_score += rewards[to] - distance_matrix[from][to];
    }
    total_score += rewards[route[0]] - distance_matrix[route[route.len() - 1]][route[0]];

    total_score
}

fn dump_solution(filename: &str, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, route: &Vec<usize>, score: i64) {
    let mut file = fs::File::options()
        .append(true)
        .create(true)
        .open(filename)
        .expect("Failed to open or create the file");

    for i in 0..route.len() {
        let from = route[i] as usize;
        writeln!(&mut file, "{}", from).expect("Failed to write to file");
    }
}

fn run_tests(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) {
    let mut rng = rand::thread_rng();
    let solvers : Vec<fn(&Vec<Vec<i64>>, &Vec<i64>, &Vec<usize>, bool) -> (Vec<usize>, i64, i64)> = vec![
        solve_random,
        solve_greedy_nn,
        solve_greedy_nna,
        solve_greedy_gc,
        solve_greedy_gca,
        solve_2_regret,
        solve_2_regret_weighted_wrapper
    ];
    let methods : Vec<&str> = vec![
        "random",
        "greedy_nn",
        "greedy_nna",
        "greedy_gc",
        "greedy_gca",
        "2_regret",
        "2_regret_weighted"
    ];

    for (idx, solver) in solvers.iter().enumerate() {
        let results : Vec<(usize, i64, i64)> = (0..distance_matrix.len()).map(|i| {
            let mut visit_subset  : Vec<usize> = vec![i];
            let mut _other_points : Vec<usize> = (0..distance_matrix.len()).filter(|&x| x != i).collect();
            _other_points.shuffle(&mut rng);
            visit_subset.extend(_other_points);

            let (route, score, length) = solver(distance_matrix, rewards, &visit_subset, false);
            // assert_eq!(score, calculate_score(&route, distance_matrix, rewards), "The calculated score does not match the expected score for method {}", methods[idx]);
            (i, score, length)
        }).collect();

        let mut dump_filename : String = "./solutions/solution_".to_owned() + methods[idx] + "_b_.csv";
        let mut file = fs::File::options()
            .append(true)
            .create(true)
            .open(dump_filename)
            .expect("Failed to open or create the file");

        writeln!(&mut file, "start;score;length").expect("Failed to write header to file");
        results.iter().for_each(|(start, score, length)| {
            writeln!(&mut file, "{};{};{}", start, score, length).expect("Failed to write to file");
        });
    }


}

fn initialize_neighborhood(route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, neighborhood_type: NeighborhoodType) -> HashMap<Move, i64> {
    let mut neighborhood : HashMap<Move, i64> = HashMap::new();
    for i in 0..route.len() {
        let prev        : usize = route[(i + route.len() - 1) % route.len()];
        let current     : usize = route[i];
        let next        : usize = route[(i + 1) % route.len()];
        let delta_score : i64 = 
            - rewards[current] 
            + distance_matrix[prev][current] 
            + distance_matrix[current][next] 
            - distance_matrix[prev][next];
        neighborhood.insert(Move::ExtRemove { idx: i }, delta_score);
    }

    let missing_nodes : Vec<usize> = (0..distance_matrix.len()).filter(|&x| !route.contains(&x)).collect();
    for &node in &missing_nodes {
        for i in 0..route.len() {
            let prev        : usize = route[i];
            let next        : usize = route[(i + 1) % route.len()];
            let delta_score : i64   = rewards[node] - distance_matrix[prev][node] - distance_matrix[node][next] + distance_matrix[prev][next];
            neighborhood.insert(Move::ExtInsert { node, after_idx: (i + 1) }, delta_score);
        }
    }
    if neighborhood_type == NeighborhoodType::VertexSwap {
        for i in 0..route.len() {
            for j in (i + 2)..route.len() {
                if i == 0 && j == route.len() - 1 { continue; }
                if j == i + 1 { continue; }

                let node1 : usize = route[i];
                let node2 : usize = route[j];

                let prev1 : usize = route[(i + route.len() - 1) % route.len()];
                let prev2 : usize = route[(j + route.len() - 1) % route.len()];

                let next1 : usize = route[(i + 1) % route.len()];
                let next2 : usize = route[(j + 1) % route.len()];

                

                let delta_score : i64 = distance_matrix[prev1][node1] + distance_matrix[node1][next1] 
                                    + distance_matrix[prev2][node2] + distance_matrix[node2][next2]
                                    - distance_matrix[prev1][node2] - distance_matrix[node2][next1]
                                    - distance_matrix[prev2][node1] - distance_matrix[node1][next2];

                neighborhood.insert(Move::IntVertexSwap { idx1: i, idx2: j }, delta_score);
            }
        }
    } else if neighborhood_type == NeighborhoodType::EdgeSwap {
        for i in 0..route.len() {
            for j in (i + 2)..route.len() {
                if i == 0 && j == route.len() - 1 { continue; }

                let edge1_src = route[i];
                let edge1_dst = route[(i + 1) % route.len()];
                let edge2_src = route[j];
                let edge2_dst = route[(j + 1) % route.len()];

                let delta_score = distance_matrix[edge1_src][edge2_src] + distance_matrix[edge1_dst][edge2_dst] - distance_matrix[edge1_src][edge1_dst] - distance_matrix[edge2_src][edge2_dst];

                neighborhood.insert(Move::IntEdgeSwap { reverse_start: i + 1, reverse_end: j }, delta_score*-1);
            }
        }
    }

    neighborhood
}

// fn local_search(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, score: &mut i64, neighborhood_type: NeighborhoodType, greedy: bool) -> (Vec<usize>, i64) {
//     let mut rng = rand::thread_rng();
//     let mut improved: bool = true;
//     let mut last_score = *score;
//     while improved {
//         //println!("I am actually working, route: {:?}, score: {}, last_score: {}", route, score, last_score);
//         let neighborhood = initialize_neighborhood(route, distance_matrix, rewards, neighborhood_type);
//         improved = false;

//         let best_option = if greedy {
//             neighborhood.iter()
//                 .filter(|&(_, &delta_score)| delta_score > 0)
//                 .choose(&mut rand::thread_rng())
//         } else {
//             neighborhood.iter().filter(|&(_, &delta_score)| delta_score > 0).max_by_key(|&(_, delta_score)| *delta_score)
//         };

//         if let Some((&best_move, &delta_score)) = best_option {
//             if delta_score > 0 {
//                 *score += delta_score;
//                 last_score = *score;
//                 improved = true;
//                 match best_move {
//                     Move::ExtInsert { node, after_idx } => {
//                         route.insert(after_idx , node);
//                     },
//                     Move::ExtRemove { idx } => {
//                         route.remove(idx);
//                     },
//                     Move::IntVertexSwap { idx1, idx2 } => {
//                         route.swap(idx1, idx2);
//                     },
//                     Move::IntEdgeSwap { reverse_start, reverse_end } => {
//                         route[reverse_start..=reverse_end].reverse();
//                     }
//                 }
//             }
//         }
//     }
    
//     (route.clone(), *score)
// }

fn evaluate_move(m: Move, route: &[usize], distance_matrix: &[Vec<i64>], rewards: &[i64]) -> i64 {
    let len = route.len();

    match m {
        Move::ExtRemove { idx } => {
            let prev = route[(idx + len - 1) % len];
            let current = route[idx];
            let next = route[(idx + 1) % len];
            distance_matrix[prev][current] + distance_matrix[current][next] - distance_matrix[prev][next] - rewards[current]
        },
        Move::ExtInsert { node, after_idx } => {
            let prev = route[after_idx];
            let next = route[(after_idx + 1) % len];
            rewards[node] - distance_matrix[prev][node] - distance_matrix[node][next] + distance_matrix[prev][next]
        },
        Move::IntVertexSwap { idx1, idx2 } => {
            let node1 = route[idx1];
            let node2 = route[idx2];
            
            let prev1 = route[(idx1 + len - 1) % len];
            let next2 = route[(idx2 + 1) % len];

            if idx2 == idx1 + 1 {
                distance_matrix[prev1][node1] + distance_matrix[node1][node2] + distance_matrix[node2][next2]
            - distance_matrix[prev1][node2] - distance_matrix[node2][node1] - distance_matrix[node1][next2]
            } else {
                let prev2 = route[(idx2 + len - 1) % len];
                let next1 = route[(idx1 + 1) % len];

                distance_matrix[prev1][node1] + distance_matrix[node1][next1] 
            + distance_matrix[prev2][node2] + distance_matrix[node2][next2]
            - distance_matrix[prev1][node2] - distance_matrix[node2][next1]
            - distance_matrix[prev2][node1] - distance_matrix[node1][next2]
            }
        },
        Move::IntEdgeSwap { reverse_start, reverse_end } => {
            let edge1_src = route[(reverse_start + len - 1) % len];
            let edge1_dst = route[reverse_start];
            let edge2_src = route[reverse_end];
            let edge2_dst = route[(reverse_end + 1) % len];

            distance_matrix[edge1_src][edge1_dst] + distance_matrix[edge2_src][edge2_dst]
          - distance_matrix[edge1_src][edge2_src] - distance_matrix[edge1_dst][edge2_dst]
        }
    }
}


fn local_search(
    route: &mut Vec<usize>, 
    distance_matrix: &[Vec<i64>], 
    rewards: &[i64], 
    score: &mut i64, 
    neighborhood_type: NeighborhoodType,
    greedy: bool
) -> (Vec<usize>, i64) {
    let mut rng = rand::thread_rng();
    let mut improved = true;
    
    while improved {
        improved = false;
        let len = route.len();

        let mut best_move: Option<Move> = None;
        let mut best_delta: i64 = 0;

        if greedy {
            let mut move_categories = vec![0, 1]; 
            match neighborhood_type {
                NeighborhoodType::VertexSwap => move_categories.push(2),
                NeighborhoodType::EdgeSwap => move_categories.push(3),
            }
            move_categories.shuffle(&mut rng);

            let mut route_indices: Vec<usize> = (0..len).collect();
            route_indices.shuffle(&mut rng);

            'greedy_search: for &category in &move_categories {
                match category {
                    0 => { // ExtRemove
                        for &idx in &route_indices {
                            let m = Move::ExtRemove { idx };
                            let delta = evaluate_move(m, route, distance_matrix, rewards);
                            if delta > 0 { best_move = Some(m); best_delta = delta; break 'greedy_search; }
                        }
                    },
                    1 => { // ExtInsert
                        let mut missing_nodes: Vec<usize> = (0..distance_matrix.len()).filter(|&x| !route.contains(&x)).collect();
                        missing_nodes.shuffle(&mut rng);
                        for &node in &missing_nodes {
                            for &i in &route_indices {
                                let m = Move::ExtInsert { node, after_idx: i };
                                let delta = evaluate_move(m, route, distance_matrix, rewards);
                                if delta > 0 { best_move = Some(m); best_delta = delta; break 'greedy_search; }
                            }
                        }
                    },
                   2 => { // IntVertexSwap
                        for &i in &route_indices {
                            for j in (i + 1)..len { // Starts at i + 1
                                if i == 0 && j == len - 1 { continue; } // Still need the wrap-around guard!
                                let m = Move::IntVertexSwap { idx1: i, idx2: j };
                                let delta = evaluate_move(m, route, distance_matrix, rewards);
                                if delta > 0 { best_move = Some(m); best_delta = delta; break 'greedy_search; }
                            }
                        }
                    },
                    3 => { // IntEdgeSwap
                        for &i in &route_indices {
                            for j in (i + 2)..len { // Starts at i + 2 to prevent pointless 1-node reversals
                                if i == 0 && j == len - 1 { continue; }
                                let m = Move::IntEdgeSwap { reverse_start: i + 1, reverse_end: j };
                                let delta = evaluate_move(m, route, distance_matrix, rewards);
                                if delta > 0 { best_move = Some(m); best_delta = delta; break 'greedy_search; }
                            }
                        }
                    },
                    _ => unreachable!()
                }
            }
        } else {
            for idx in 0..len {
                let m = Move::ExtRemove { idx };
                let delta = evaluate_move(m, route, distance_matrix, rewards);
                if delta > best_delta { best_move = Some(m); best_delta = delta; }
            }

            let mut in_route = vec![false; distance_matrix.len()];
            for &node in route.iter() { in_route[node] = true; }
            
            for node in 0..distance_matrix.len() {
                if !in_route[node] {
                    for i in 0..len {
                        let m = Move::ExtInsert { node, after_idx: i };
                        let delta = evaluate_move(m, route, distance_matrix, rewards);
                        if delta > best_delta { best_move = Some(m); best_delta = delta; }
                    }
                }
            }
            for i in 0..len {
                for j in (i + 2)..len {
                    if i == 0 && j == len - 1 { continue; }
                    
                    let m = match neighborhood_type {
                        NeighborhoodType::VertexSwap => Move::IntVertexSwap { idx1: i, idx2: j },
                        NeighborhoodType::EdgeSwap => Move::IntEdgeSwap { reverse_start: i + 1, reverse_end: j },
                    };

                    let delta = evaluate_move(m, route, distance_matrix, rewards);
                    if delta > best_delta { best_move = Some(m); best_delta = delta; }
                }
            }
        }
        if let Some(m) = best_move {
            *score += best_delta;
            improved = true;

            match m {
                Move::ExtInsert { node, after_idx } => { 
                    route.insert(after_idx + 1, node); 
                },
                Move::ExtRemove { idx } => { route.remove(idx); },
                Move::IntVertexSwap { idx1, idx2 } => { route.swap(idx1, idx2); },
                Move::IntEdgeSwap { reverse_start, reverse_end } => { route[reverse_start..=reverse_end].reverse(); }
            }
        }
    }
    (route.clone(), *score)
}


fn random_search(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, score: &mut i64, neighborhood_type: NeighborhoodType, time_limit: i64) -> (Vec<usize>, i64) {
    let start_time = time::Instant::now();
    let mut rng = rand::thread_rng();
    let mut best_route = route.clone();
    let mut best_score = *score;
    while start_time.elapsed().as_millis() < time_limit as u128 {
        let neighborhood = initialize_neighborhood(route, distance_matrix, rewards, neighborhood_type);
        if neighborhood.is_empty() { break; }
        let (&random_move, &delta_score) = neighborhood.iter().choose(&mut rng).unwrap();
        *score += delta_score;
        match random_move {
            Move::ExtInsert { node, after_idx } => {
                route.insert(after_idx, node);
            },
            Move::ExtRemove { idx } => {
                route.remove(idx);
            },
            Move::IntVertexSwap { idx1, idx2 } => {
                route.swap(idx1, idx2);
            },
            Move::IntEdgeSwap { reverse_start, reverse_end } => {
                route[reverse_start..=reverse_end].reverse();
            }
        }
        if *score > best_score {
            best_score = *score;
            best_route = route.clone();
        }
    }
    (best_route, best_score)
}

fn run_search_tests(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, iterations: usize) {
    let mut rng = rand::thread_rng();

    let neighborhood_types = vec![NeighborhoodType::VertexSwap, NeighborhoodType::EdgeSwap];
    let greedy_options = vec![true, false];
    let solvers: Vec<fn(&Vec<Vec<i64>>, &Vec<i64>, &Vec<usize>, bool) -> (Vec<usize>, i64, i64)> = vec![
        solve_random,
        solve_2_regret,
    ];
    let methods: Vec<&str> = vec![
        "random",
        "2_regret",
    ];

    let mut visit_subset: Vec<usize> = (0..distance_matrix.len()).collect();
    
    visit_subset.shuffle(&mut rng);
    let mut max_found_time: i64 = 0;

    for neighborhood_type in neighborhood_types.iter() {
        for &greedy in &greedy_options {
            for (idx, solver) in solvers.iter().enumerate() {
                let mut min_score: i64 = i64::MAX;
                let mut max_score: i64 = i64::MIN;
                let mut best_solution : Option<(Vec<usize>, i64)> = None;
                let mut sum_score: i64 = 0;
                let mut times: Vec<i64> = Vec::with_capacity(iterations);

                let dump_filename: String = "./solutions/solution_".to_owned()
                    + "local_search_"
                    + &greedy.to_string()
                    + "_"
                    + &neighborhood_type.to_string()
                    + "_"
                    + methods[idx]
                    + "_b_.csv";

                let mut file = fs::File::options()
                    .append(true)
                    .create(true)
                    .open(dump_filename)
                    .expect("Failed to open or create the file");

                writeln!(&mut file, "iteration,score").unwrap();

                for iter in 0..iterations {
                    visit_subset.shuffle(&mut rng);
                    let (mut route, mut score, _) =
                        solver(&distance_matrix, &rewards, &visit_subset, true);

                    let iter_start_time = time::Instant::now();
                    //println!("Started");
                    let (optimized_route, optimized_score) = local_search(
                        &mut route.clone(),
                        &distance_matrix,
                        &rewards,
                        &mut score.clone(),
                        *neighborhood_type,
                        greedy,
                    );
                    //println!("Stuff finished");
                    let iter_elapsed_time = iter_start_time.elapsed().as_millis() as i64;
                    times.push(iter_elapsed_time);
                    if iter_elapsed_time > max_found_time {
                        max_found_time = iter_elapsed_time;
                    }

                    min_score = cmp::min(min_score, optimized_score);
                    max_score = cmp::max(max_score, optimized_score);
                    sum_score += optimized_score;
                    if (optimized_score == max_score) {
                        best_solution = Some((optimized_route.clone(), optimized_score));
                    }

                    writeln!(&mut file, "{},{}", iter, optimized_score).unwrap();

                    assert_eq!(
                        optimized_score,
                        calculate_score(&optimized_route, distance_matrix, rewards),
                        "The calculated score does not match the expected score for method {}, neighborhood {:?}, iteration {}",
                        methods[idx],
                        &neighborhood_type.to_string(),
                        iter
                    );
                }

                // Time statistics (milliseconds)
                let min_time = times.iter().min().cloned().unwrap_or(0);
                let max_time = times.iter().max().cloned().unwrap_or(0);
                let avg_time = if !times.is_empty() {
                    times.iter().sum::<i64>() / times.len() as i64
                } else { 0 };

                writeln!(&mut file, "average,min_score,max_score,min_time_ms,max_time_ms,avg_time_ms").unwrap();
                writeln!(
                    &mut file,
                    "{},{},{},{},{},{}",
                    sum_score / iterations as i64,
                    min_score,
                    max_score,
                    min_time,
                    max_time,
                    avg_time
                ).unwrap();
                if let Some((ref best_route, best_score)) = best_solution {
                    assert_eq!(best_score, calculate_score(best_route, &distance_matrix, &rewards), "Best solution score does not match calculated score!");
                }
                writeln!(&mut file, "best_solution,{:?}", best_solution.clone().unwrap());
            }
        }
    }

    for neighborhood_type in neighborhood_types.iter() {
        for (idx, solver) in solvers.iter().enumerate() {
            let mut min_score: i64 = i64::MAX;
            let mut max_score: i64 = i64::MIN;
            let mut best_solution : Option<(Vec<usize>, i64)> = None;
            let mut sum_score: i64 = 0;
            let mut times: Vec<i64> = Vec::with_capacity(iterations);

            let dump_filename: String = "./solutions/solution_".to_owned()
                + "random_search_"
                + &neighborhood_type.to_string()
                + "_"
                + methods[idx]
                + "_b_.csv";

            let mut file = fs::File::options()
                .append(true)
                .create(true)
                .open(dump_filename)
                .expect("Failed to open or create the file");

            writeln!(&mut file, "iteration,score").unwrap();

            for iter in 0..iterations {
                visit_subset.shuffle(&mut rng);
                let (mut route, mut score, _) =
                    solver(distance_matrix, rewards, &visit_subset, true);

                let iter_start_time = time::Instant::now();
                let (optimized_route, optimized_score) = random_search(
                    &mut route.clone(),
                    distance_matrix,
                    rewards,
                    &mut score.clone(),
                    *neighborhood_type,
                    max_found_time,
                );
                let iter_elapsed_time = iter_start_time.elapsed().as_millis() as i64;
                times.push(iter_elapsed_time);

                min_score = min_score.min(optimized_score);
                max_score = max_score.max(optimized_score);
                sum_score += optimized_score;
                if (optimized_score == max_score) {
                    best_solution = Some((optimized_route.clone(), optimized_score));
                }

                writeln!(&mut file, "{},{}", iter, optimized_score).unwrap();

                assert_eq!(
                    optimized_score,
                    calculate_score(&optimized_route, distance_matrix, rewards),
                    "The calculated score does not match the expected score for method {}, neighborhood {:?}, iteration {}",
                    methods[idx],
                    &neighborhood_type.to_string(),
                    iter
                );
            }

            // Time statistics (milliseconds)
            let min_time = times.iter().min().cloned().unwrap_or(0);
            let max_time = times.iter().max().cloned().unwrap_or(0);
            let avg_time = if !times.is_empty() {
                times.iter().sum::<i64>() / times.len() as i64
            } else { 0 };

            writeln!(&mut file, "average,min_score,max_score,min_time_ms,max_time_ms,avg_time_ms").unwrap();
            writeln!(
                &mut file,
                "{},{},{},{},{},{}",
                sum_score / iterations as i64,
                min_score,
                max_score,
                min_time,
                max_time,
                avg_time
            ).unwrap();
            if let Some((ref best_route, best_score)) = best_solution {
                assert_eq!(best_score, calculate_score(best_route, &distance_matrix, &rewards), "Best solution score does not match calculated score!");
            }
            writeln!(&mut file, "best_solution,{:?}", best_solution.clone().unwrap());
        }
    }
}

fn solve_candidate_moves(route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> (Vec<usize>, i64){
    let k: i32 = 10;
    let mut found_better: bool = true;
    let mut score = calculate_score(route, distance_matrix, rewards);
    let mut current_route = route.clone();
    while found_better{
        found_better = false;
        let n = current_route.len();
        let mut all_candidate_moves: Vec<(Move, i64)> = Vec::new();
        for i in (0..n) {
            let current_node = current_route[i];
            let closest_nodes: Vec<(usize, i64)> = (0..distance_matrix.len())
                .filter(|&x| x != current_node)
                .map(|x| (x, distance_matrix[current_node][x]))
                .collect();
            let k_closest_nodes: Vec<(usize, i64)> = closest_nodes
                .iter()
                .cloned()
                .filter(|(node, _)| !current_route.contains(node))
                .take(k as usize)
                .collect();
            for (candidate_node, _) in k_closest_nodes {
                if(!current_route.contains(&candidate_node)){
                    let delta_score = rewards[candidate_node] - distance_matrix[current_node][candidate_node] - distance_matrix[candidate_node][current_route[(i + 1) % n]] + distance_matrix[current_node][current_route[(i + 1) % n]];
                    all_candidate_moves.push((Move::ExtInsert { node: candidate_node, after_idx: i + 1 }, delta_score));
                }
                else{
                    let candidate_idx = current_route.iter().position(|&x| x == candidate_node).unwrap();
                    if candidate_idx == (i + 1) % n{
                        break;
                    }
                    //We only do edge swaps.
                    let delta_score = distance_matrix[current_node][candidate_node] + distance_matrix[current_route[(i + 1) % n]][current_route[(candidate_idx + 1) % n]]
                                - distance_matrix[current_node][current_route[(i + 1) % n]] - distance_matrix[candidate_node][current_route[(candidate_idx + 1) % n]];
                    all_candidate_moves.push((Move::IntEdgeSwap { reverse_start: i, reverse_end: candidate_idx }, -1*delta_score));

                    //Remove and connect to candidate node
                    let delta_score_remove = -rewards[current_node] + distance_matrix[current_route[(i + n - 1) % n]][current_route[(i + 1) % n]] - distance_matrix[current_route[(i + n - 1) % n]][current_node] - distance_matrix[current_node][current_route[(i + 1) % n]];
                    all_candidate_moves.push((Move::ExtRemove { idx: i }, delta_score_remove));
                }
            }
            all_candidate_moves.sort_by_key(|&(_, delta)| -delta);
            let best_move = all_candidate_moves.first().unwrap();
            if best_move.1 > 0 {
                score += best_move.1;
                match best_move.0 {
                    Move::ExtInsert { node, after_idx } => {
                        current_route.insert(after_idx, node);
                    },
                    Move::ExtRemove { idx } => {
                        current_route.remove(idx);
                    },
                    Move::IntVertexSwap { idx1, idx2 } => {
                        current_route.swap(idx1, idx2);
                    },
                    Move::IntEdgeSwap { reverse_start, reverse_end } => {
                        current_route[reverse_start..=reverse_end].reverse();
                    }
                }
                found_better = true;
                break;
            }
            
        }
    }
    (current_route, score)
}

fn check_if_move_is_legal(route: &Vec<usize>, next_move: &Move) -> bool {
    match next_move {
        Move::ExtInsert { node, after_idx } => {
            !route.contains(node) && *after_idx <= route.len()
        },
        Move::ExtRemove { idx } => {
            *idx < route.len()
        },
        Move::IntVertexSwap { idx1, idx2 } => {
            *idx1 < route.len() && *idx2 < route.len() && idx1 != idx2
        },
        Move::IntEdgeSwap { reverse_start, reverse_end } => {
            *reverse_start < route.len() && *reverse_end < route.len() && reverse_start != reverse_end
        }
    }
}

fn recalculate_deltas(
    route: &Vec<usize>,
    moves: &[(Move, i64)],
    distance_matrix: &Vec<Vec<i64>>,
    rewards: &Vec<i64>,
) -> VecDeque<(Move, i64)> {
    let route_set: HashSet<usize> = route.iter().cloned().collect();
    let mut updated: Vec<(Move, i64)> = moves
        .iter()
        .filter_map(|(m, _)| {
            let delta = match *m {
                Move::ExtRemove { idx } => {
                    if idx >= route.len() { return None; }
                    let prev    = route[(idx + route.len() - 1) % route.len()];
                    let current = route[idx];
                    let next    = route[(idx + 1) % route.len()];
                    - rewards[current]
                    + distance_matrix[prev][current]
                    + distance_matrix[current][next]
                    - distance_matrix[prev][next]
                },
                Move::ExtInsert { node, after_idx } => {
                    if route_set.contains(&node) || after_idx > route.len() { return None; }
                    let prev = route[(after_idx + route.len() - 1) % route.len()];
                    let next = route[after_idx % route.len()];
                    rewards[node]
                    - distance_matrix[prev][node]
                    - distance_matrix[node][next]
                    + distance_matrix[prev][next]
                },
                Move::IntVertexSwap { idx1, idx2 } => {
                    if idx1 >= route.len() || idx2 >= route.len() { return None; }
                    let node1 = route[idx1];
                    let node2 = route[idx2];
                    let prev1 = route[(idx1 + route.len() - 1) % route.len()];
                    let next1 = route[(idx1 + 1) % route.len()];
                    let prev2 = route[(idx2 + route.len() - 1) % route.len()];
                    let next2 = route[(idx2 + 1) % route.len()];
                    distance_matrix[prev1][node1] + distance_matrix[node1][next1]
                    + distance_matrix[prev2][node2] + distance_matrix[node2][next2]
                    - distance_matrix[prev1][node2] - distance_matrix[node2][next1]
                    - distance_matrix[prev2][node1] - distance_matrix[node1][next2]
                },
                Move::IntEdgeSwap { reverse_start, reverse_end } => {
                if reverse_start >= route.len() || reverse_end >= route.len() { return None; }
                let n = route.len();
                let edge1_src = route[(reverse_start + n - 1) % n]; 
                let edge1_dst = route[reverse_start];
                let edge2_src = route[reverse_end];
                let edge2_dst = route[(reverse_end + 1) % n];
                distance_matrix[edge1_src][edge1_dst] + distance_matrix[edge2_src][edge2_dst]
                - (distance_matrix[edge1_src][edge2_src] + distance_matrix[edge1_dst][edge2_dst])
            },
            };
            if delta > 0 { Some((*m, delta)) } else { None }
        })
        .collect();

    updated.sort_by_key(|&(_, d)| -d);
    updated.into_iter().collect()
}

fn solve_lm_moves(route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> (Vec<usize>, i64) {
    let mut current_route = route.clone();
    let mut score = calculate_score(&current_route, distance_matrix, rewards);
    let mut lm: VecDeque<(Move, i64)> = VecDeque::new();
    let neighborhood = initialize_neighborhood(&current_route, distance_matrix, rewards, NeighborhoodType::EdgeSwap);
    let mut moves: Vec<(Move, i64)> = neighborhood
            .iter()
            .filter(|(_, &delta)| delta > 0)
            .map(|(m, &d)| (*m, d))
            .collect();

    moves.sort_by_key(|&(_, d)| -d);

    lm.extend(moves.into_iter().map(|(m, d)| (m, d)));
    while !lm.is_empty(){
        let mut is_applied = false;
        for _ in 0..lm.len(){
            let (next_move, delta_score) = lm.pop_front().unwrap();
            if check_if_move_is_legal(&current_route, &next_move) {
                if delta_score <= 0{
                    break
                }
                match next_move {
                    Move::ExtInsert { node, after_idx } => { current_route.insert(after_idx, node); },
                    Move::ExtRemove { idx }             => { current_route.remove(idx); },
                    Move::IntVertexSwap { idx1, idx2 } => { current_route.swap(idx1, idx2); },
                    Move::IntEdgeSwap { reverse_start, reverse_end } => {
                        current_route[reverse_start..=reverse_end].reverse();
                    }
                }
                is_applied = true;
                score += delta_score;
                let remaining: Vec<(Move, i64)> = lm.drain(..).collect();
                lm = recalculate_deltas(&current_route, &remaining, distance_matrix, rewards);
                break;
            }
        }
        if !is_applied {
            lm.clear();
            let new_neighborhood = initialize_neighborhood(&current_route, distance_matrix, rewards, NeighborhoodType::EdgeSwap);
            let mut new_moves: Vec<(Move, i64)> = new_neighborhood
                .iter()
                .filter(|(_, &delta)| delta > 0)
                .map(|(m, &d)| (*m, d))
                .collect();
            new_moves.sort_by_key(|&(_, d)| -d);
            lm.extend(new_moves.into_iter().map(|(m, d)| (m, d)));
        }
    }
    (current_route, score)
}

fn run_candidate_tests(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, iterations: usize) {
    use std::time::Instant;
    use rand::seq::SliceRandom;
    let n = distance_matrix.len();
    let iterations = iterations;
    let mut rng = rand::thread_rng();

    let methods = vec![
        ("lm", solve_lm_moves as fn(&Vec<usize>, &Vec<Vec<i64>>, &Vec<i64>) -> (Vec<usize>, i64)),
        ("candidate", solve_candidate_moves as fn(&Vec<usize>, &Vec<Vec<i64>>, &Vec<i64>) -> (Vec<usize>, i64)),
    ];

    for (method_name, solver) in methods {
        let mut scores = Vec::with_capacity(iterations);
        let mut times = Vec::with_capacity(iterations);
        let mut best_solution: Option<(Vec<usize>, i64)> = None;
        let mut best_score = i64::MIN;

        let filename = format!("./solutions/solution_{}_b_.csv", method_name);
        let mut file = std::fs::File::options()
            .append(true)
            .create(true)
            .open(&filename)
            .expect("Failed to open or create the file");
        writeln!(&mut file, "iteration,score,time_ms").unwrap();

        for iter in 0..iterations {
            let mut visit_subset: Vec<usize> = (0..distance_matrix.len()).collect();
            visit_subset.shuffle(&mut rng);
            let (route, _, _) = solve_random(&distance_matrix, &rewards, &visit_subset, true);
            let start_time = Instant::now();
            let (sol, score) = solver(&route, distance_matrix, rewards);
            let elapsed = start_time.elapsed().as_nanos() as i64;
            scores.push(score);
            times.push(elapsed);
            if score > best_score {
                best_score = score;
                best_solution = Some((sol.clone(), score));
            }
            assert_eq!(score, calculate_score(&sol, distance_matrix, rewards), "Calculated score does not match expected score for method {}, iteration {}", method_name, iter);
            writeln!(&mut file, "{},{},{}", iter, score, elapsed).unwrap();
        }

        let avg_score = scores.iter().sum::<i64>() as f64 / scores.len() as f64;
        let min_score = *scores.iter().min().unwrap_or(&0);
        let max_score = *scores.iter().max().unwrap_or(&0);
        let avg_time = times.iter().sum::<i64>() as f64 / times.len() as f64;
        let min_time = *times.iter().min().unwrap_or(&0);
        let max_time = *times.iter().max().unwrap_or(&0);

        writeln!(&mut file, "avg_score,min_score,max_score,avg_time_ms,min_time_ms,max_time_ms").unwrap();
        writeln!(
            &mut file,
            "{:.2},{},{},{:.2},{},{}",
            avg_score, min_score, max_score, avg_time, min_time, max_time
        ).unwrap();
        if let Some((ref best_route, best_score)) = best_solution {
            writeln!(&mut file, "best_solution,{:?},{}", best_route, best_score).unwrap();
        }
    }
}

fn run_msls(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, neighborhood_type: NeighborhoodType, greedy: bool) -> (Vec<usize>, i64) {
    let mut rng = rand::thread_rng();
    let mut best_route: Vec<usize> = Vec::new();
    let mut best_score = i64::MIN;

    for _ in 0..200 {
        let mut visit_subset: Vec<usize> = (0..distance_matrix.len()).collect();
        visit_subset.shuffle(&mut rng);

        let (mut route, mut score, _) = solve_random(distance_matrix, rewards, &visit_subset, false);

        let (optimized_route, optimized_score) = local_search(
            &mut route,
            distance_matrix,
            rewards,
            &mut score,
            neighborhood_type,
            greedy,
        );

        if optimized_score > best_score {
            best_score = optimized_score;
            best_route = optimized_route;
        }
    }

    (best_route, best_score)
}

fn run_ils(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, neighborhood_type: NeighborhoodType, greedy: bool, time_limit: std::time::Duration) -> (Vec<usize>, i64){
    let (mut start_route, mut start_score, _) = solve_random(distance_matrix, rewards, &(0..distance_matrix.len()).collect(), false);
    let mut final_route = start_route.clone();
    let mut final_score = start_score;

    (start_route, start_score) = local_search(
        &mut start_route.clone(),
        distance_matrix,
        rewards,
        &mut start_score.clone(),
        neighborhood_type,
        greedy,
    );

    if start_score > final_score {
        final_score = start_score;
        final_route = start_route.clone();
    }
    
    let mut start_time = std::time::Instant::now();
    while start_time.elapsed() < time_limit {
        let (peturbated_route, peturbated_score) = create_peturbations(&start_route, distance_matrix, start_score, rewards, neighborhood_type);
        let (optimized_route, optimized_score) = local_search(
            &mut peturbated_route.clone(),
            distance_matrix,
            rewards,
            &mut peturbated_score.clone(),
            neighborhood_type,
            greedy,
        );
        if optimized_score > final_score {
            final_score = optimized_score;
            final_route = optimized_route.clone();
            start_route = optimized_route;
            start_score = optimized_score;
        }
    }
    (final_route, final_score)
}

fn create_peturbations(current_route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, current_score: i64, rewards: &Vec<i64>, neighborhood_type: NeighborhoodType) -> (Vec<usize>, i64) {
    let n = current_route.len();
    let mut puteurbated_route = current_route.clone();
    let mut delta_score = current_score;
    //do random 5 moves of the given neighborhood type
    let mut rng = rand::thread_rng();
    for _ in 0..5 {
        match neighborhood_type {
            NeighborhoodType::VertexSwap => {
                let idx1 = rng.gen_range(0..n);
                let idx2 = rng.gen_range(0..n);
                if idx1 == idx2 {
                    continue;
                }
                let node1 = puteurbated_route[idx1];
                let node2 = puteurbated_route[idx2];
                let prev1 = puteurbated_route[(idx1 + n - 1) % n];
                let next1 = puteurbated_route[(idx1 + 1) % n];
                let prev2 = puteurbated_route[(idx2 + n - 1) % n];
                let next2 = puteurbated_route[(idx2 + 1) % n];
                delta_score -= distance_matrix[prev1][node1] + distance_matrix[node1][next1] 
                                    + distance_matrix[prev2][node2] + distance_matrix[node2][next2]
                                    - distance_matrix[prev1][node2] - distance_matrix[node2][next1]
                                    - distance_matrix[prev2][node1] - distance_matrix[node1][next2];
                puteurbated_route.swap(idx1, idx2);
            },
            NeighborhoodType::EdgeSwap => {
                let idx1 = rng.gen_range(0..n);
                let idx2 = rng.gen_range(0..n);
                if idx1 != idx2 {
                    let (start, end) = if idx1 < idx2 { (idx1, idx2) } else { (idx2, idx1) };
                    let edge1_src = puteurbated_route[(start + n - 1) % n];
                    let edge1_dst = puteurbated_route[start];
                    let edge2_src = puteurbated_route[(end + n - 1) % n];
                    let edge2_dst = puteurbated_route[end];
                    delta_score -= distance_matrix[edge1_src][edge2_src] + distance_matrix[edge1_dst][edge2_dst] - distance_matrix[edge1_src][edge1_dst] - distance_matrix[edge2_src][edge2_dst];
                    puteurbated_route[start..=end].reverse();
                }
            },
        }
    }
    let real_score = calculate_score(&puteurbated_route, distance_matrix, rewards);
    (puteurbated_route, real_score)

}

fn run_lns(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, neighborhood_type: NeighborhoodType, greedy: bool, time_limit: std::time::Duration) -> (Vec<usize>, i64){
        let (mut start_route, mut start_score, _) = solve_random(distance_matrix, rewards, &(0..distance_matrix.len()).collect(), false);
        let mut final_route = start_route.clone();
        let mut final_score = start_score;
    
        (start_route, start_score) = local_search(
            &mut start_route.clone(),
            distance_matrix,
            rewards,
            &mut start_score.clone(),
            neighborhood_type,
            greedy,
        );
    
        if start_score > final_score {
            final_score = start_score;
            final_route = start_route.clone();
        }
        
        let mut start_time = std::time::Instant::now();
        while start_time.elapsed() < time_limit {
            let (destroyed_route, destroyed_score) = destroy_route(&start_route, distance_matrix, rewards, start_score, neighborhood_type);
            let (fixed_route, fixed_score) = fix_route(&destroyed_route, distance_matrix, rewards);
            let (optimized_route, optimized_score) = local_search(
                &mut fixed_route.clone(),
                distance_matrix,
                rewards,
                &mut fixed_score.clone(),
                neighborhood_type,
                greedy,
            );
            if optimized_score > final_score {
                final_score = optimized_score;
                final_route = optimized_route.clone();
                start_route = optimized_route;
                start_score = optimized_score;
            }
        }
        (final_route, final_score)
}

fn destroy_route(current_route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, current_score: i64, neighborhood_type: NeighborhoodType) -> (Vec<usize>, i64){
    //destroy 30% 
    let route_length = current_route.len();
    let mut left_to_destroy = (route_length as f64 * 0.3).round() as usize;
    let mut destroyed_route = current_route.clone();
    let mut rng = rand::thread_rng();
    while left_to_destroy > 0 {
        let move_choose = rng.gen_range(0..2);
        match move_choose {
            0 => {
                //remove a random node
                let idx = rng.gen_range(0..destroyed_route.len());
                destroyed_route.remove(idx);
                left_to_destroy -= 1;
            },
            1 => {
                //remove a random edge
                let idx1 = rng.gen_range(0..destroyed_route.len());
                let idx2 = (idx1 + 1) % destroyed_route.len();
                destroyed_route.remove(idx1);
                left_to_destroy -= 1;
            }
            _ => {}
        }
    }
    let destroyed_score = calculate_score(&destroyed_route, distance_matrix, rewards);
    (destroyed_route, destroyed_score)
}

fn fix_route(destroyed_route: &Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> (Vec<usize>, i64){
    let mut fixed_route = destroyed_route.clone();

    let mut route = destroyed_route.clone();
    let n = rewards.len();
    let mut in_route = vec![false; n];
    for &node in &route {
        in_route[node] = true;
    }

    let mut candidates: Vec<usize> = (0..n).filter(|&i| !in_route[i]).collect();

    let mut total_score: i64 = 0;

    if route.len() > 1 {
        for i in 0..route.len() {
            let from = route[i];
            let to   = route[(i + 1) % route.len()];
            total_score += rewards[from] - distance_matrix[from][to];
        }
    } else if route.len() == 1 {
        total_score += rewards[route[0]];
    }

    while !candidates.is_empty() {
        let mut best_regret = i64::MIN;
        let mut best_point  = 0;
        let mut best_cost   = 0;
        let mut best_pos    = 0;
        let mut best_idx    = 0;

        for (idx, &point) in candidates.iter().enumerate() {
            let mut best_insert = i64::MAX;
            let mut second_best = i64::MAX;
            let mut insert_pos  = 0;

            for i in 0..route.len() {
                let from = route[i];
                let to   = route[(i + 1) % route.len()];

                let cost = distance_matrix[from][point]
                         + distance_matrix[point][to]
                         - distance_matrix[from][to];

                if cost < best_insert {
                    second_best = best_insert;
                    best_insert = cost;
                    insert_pos = i + 1;
                } else if cost < second_best {
                    second_best = cost;
                }
            }

            let regret = second_best - best_insert;

            if regret > best_regret {
                best_regret = regret;
                best_point  = point;
                best_cost   = best_insert;
                best_pos    = insert_pos;
                best_idx    = idx;
            }
        }

        route.insert(best_pos, best_point);
        total_score += rewards[best_point] - best_cost;

        candidates.swap_remove(best_idx);
    }

    let (route, total_score, _) = solve_greedy_phase2(distance_matrix, rewards, &mut route, 0, total_score);

    (route, total_score)
}

fn run_extended_tests(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) {
    use std::time::Instant;
    use rand::seq::SliceRandom;
    let iterations = 20;
    let mut rng = rand::thread_rng();
    let methods = vec!["msls", "ils", "lns"];
    let neighborhood_type = NeighborhoodType::EdgeSwap;
    let greedy = false;

    // MSLS
    let mut msls_scores = Vec::with_capacity(iterations);
    let mut msls_times = Vec::with_capacity(iterations);
    let mut msls_best_solution: Option<(Vec<usize>, i64)> = None;
    let mut msls_best_score = i64::MIN;

    let msls_filename = "./solutions/solution_msls_b_.csv";
    let mut msls_file = std::fs::File::options()
        .append(true)
        .create(true)
        .open(&msls_filename)
        .expect("Failed to open or create the file");
    writeln!(&mut msls_file, "iteration,score,time_ms").unwrap();

    let mut msls_max_time = 0i64;
    for iter in 0..2 {
        let start_time = Instant::now();
        let (route, score) = run_msls(distance_matrix, rewards, neighborhood_type, greedy);
        let elapsed = start_time.elapsed().as_millis() as i64;
        if elapsed > msls_max_time { msls_max_time = elapsed; }
        msls_scores.push(score);
        msls_times.push(elapsed);
        if score > msls_best_score {
            msls_best_score = score;
            msls_best_solution = Some((route.clone(), score));
        }
        writeln!(&mut msls_file, "{},{},{}", iter, score, elapsed).unwrap();
    }

    let msls_avg_score = msls_scores.iter().sum::<i64>() as f64 / msls_scores.len() as f64;
    let msls_min_score = *msls_scores.iter().min().unwrap_or(&0);
    let msls_max_score = *msls_scores.iter().max().unwrap_or(&0);
    let msls_avg_time = msls_times.iter().sum::<i64>() as f64 / msls_times.len() as f64;
    let msls_min_time = *msls_times.iter().min().unwrap_or(&0);
    let msls_max_time_stat = *msls_times.iter().max().unwrap_or(&0);
    writeln!(&mut msls_file, "avg_score,min_score,max_score,avg_time_ms,min_time_ms,max_time_ms").unwrap();
    writeln!(
        &mut msls_file,
        "{:.2},{},{},{:.2},{},{}",
        msls_avg_score, msls_min_score, msls_max_score, msls_avg_time, msls_min_time, msls_max_time_stat
    ).unwrap();
    if let Some((ref best_route, best_score)) = msls_best_solution {
        writeln!(&mut msls_file, "best_solution,{:?},{}", best_route, best_score).unwrap();
    }

    // ILS
    let mut ils_scores = Vec::with_capacity(iterations);
    let mut ils_times = Vec::with_capacity(iterations);
    let mut ils_best_solution: Option<(Vec<usize>, i64)> = None;
    let mut ils_best_score = i64::MIN;

    let ils_filename = "./solutions/solution_ils_b_.csv";
    let mut ils_file = std::fs::File::options()
        .append(true)
        .create(true)
        .open(&ils_filename)
        .expect("Failed to open or create the file");
    writeln!(&mut ils_file, "iteration,score,time_ms").unwrap();

    let ils_time_limit = std::time::Duration::from_millis(msls_max_time as u64);
    for iter in 0..iterations {
        let start_time = Instant::now();
        let (route, score) = run_ils(distance_matrix, rewards, neighborhood_type, greedy, ils_time_limit);
        let elapsed = start_time.elapsed().as_millis() as i64;
        ils_scores.push(score);
        ils_times.push(elapsed);
        if score > ils_best_score {
            ils_best_score = score;
            ils_best_solution = Some((route.clone(), score));
        }
        writeln!(&mut ils_file, "{},{},{}", iter, score, elapsed).unwrap();
    }

    let ils_avg_score = ils_scores.iter().sum::<i64>() as f64 / ils_scores.len() as f64;
    let ils_min_score = *ils_scores.iter().min().unwrap_or(&0);
    let ils_max_score = *ils_scores.iter().max().unwrap_or(&0);
    let ils_avg_time = ils_times.iter().sum::<i64>() as f64 / ils_times.len() as f64;
    let ils_min_time = *ils_times.iter().min().unwrap_or(&0);
    let ils_max_time_stat = *ils_times.iter().max().unwrap_or(&0);
    writeln!(&mut ils_file, "avg_score,min_score,max_score,avg_time_ms,min_time_ms,max_time_ms").unwrap();
    writeln!(
        &mut ils_file,
        "{:.2},{},{},{:.2},{},{}",
        ils_avg_score, ils_min_score, ils_max_score, ils_avg_time, ils_min_time, ils_max_time_stat
    ).unwrap();
    if let Some((ref best_route, best_score)) = ils_best_solution {
        writeln!(&mut ils_file, "best_solution,{:?},{}", best_route, best_score).unwrap();
    }

    // LNS
    let mut lns_scores = Vec::with_capacity(iterations);
    let mut lns_times = Vec::with_capacity(iterations);
    let mut lns_best_solution: Option<(Vec<usize>, i64)> = None;
    let mut lns_best_score = i64::MIN;

    let lns_filename = "./solutions/solution_lns_b_.csv";
    let mut lns_file = std::fs::File::options()
        .append(true)
        .create(true)
        .open(&lns_filename)
        .expect("Failed to open or create the file");
    writeln!(&mut lns_file, "iteration,score,time_ms").unwrap();

    let lns_time_limit = std::time::Duration::from_millis(msls_max_time as u64);
    for iter in 0..iterations {
        let start_time = Instant::now();
        let (route, score) = run_lns(distance_matrix, rewards, neighborhood_type, greedy, lns_time_limit);
        let elapsed = start_time.elapsed().as_millis() as i64;
        lns_scores.push(score);
        lns_times.push(elapsed);
        if score > lns_best_score {
            lns_best_score = score;
            lns_best_solution = Some((route.clone(), score));
        }
        writeln!(&mut lns_file, "{},{},{}", iter, score, elapsed).unwrap();
    }

    let lns_avg_score = lns_scores.iter().sum::<i64>() as f64 / lns_scores.len() as f64;
    let lns_min_score = *lns_scores.iter().min().unwrap_or(&0);
    let lns_max_score = *lns_scores.iter().max().unwrap_or(&0);
    let lns_avg_time = lns_times.iter().sum::<i64>() as f64 / lns_times.len() as f64;
    let lns_min_time = *lns_times.iter().min().unwrap_or(&0);
    let lns_max_time_stat = *lns_times.iter().max().unwrap_or(&0);
    writeln!(&mut lns_file, "avg_score,min_score,max_score,avg_time_ms,min_time_ms,max_time_ms").unwrap();
    writeln!(
        &mut lns_file,
        "{:.2},{},{},{:.2},{},{}",
        lns_avg_score, lns_min_score, lns_max_score, lns_avg_time, lns_min_time, lns_max_time_stat
    ).unwrap();
    if let Some((ref best_route, best_score)) = lns_best_solution {
        writeln!(&mut lns_file, "best_solution,{:?},{}", best_route, best_score).unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <csv_file_path>", args[0]);
        return;
    }

    let     file_path       : &String                               = &args[1];
    let     read_result     : Result<Vec<Vec<f64>>, Box<dyn Error>> = read_csv_mapped(file_path);
    let     row_num         : usize                                 = read_result.as_ref().map(|v| v.len()).unwrap_or(0);
    let     col_num         : usize                                 = read_result.as_ref().map(|v| v[0].len()).unwrap_or(0);
    let mut distance_matrix : Vec<Vec<i64>>                         = Vec::new();
    let mut rewards         : Vec<i64>                              = Vec::new();

    if row_num == 0 || col_num == 0 {
        println!("The file is empty or not properly formatted.");
        return;
    }

    if row_num != col_num {
        println!("Assuming a set of coordinates, creating a distance matrix");
        if col_num < 2 {
            println!("The file must contain at least two columns to create a distance matrix.");
            return;
        }
        let (_distances, _rewards) = get_distance_matrix_and_rewards(read_result.unwrap());
        distance_matrix = _distances;
        rewards = _rewards;
    } else {
        println!("Assuming a distance matrix, using it directly");
        distance_matrix = (read_result.unwrap()).into_iter().map(|row| row.into_iter().map(|value| value as i64).collect()).collect();
    }

    // run_tests(&distance_matrix, &rewards); //old code - task 1

    // let mut rng = rand::thread_rng();

    // let mut max_time : i64 = 0;

    // let mut start_time = time::Instant::now();
    // let mut visit_subset: Vec<usize> = (0..distance_matrix.len()).collect();
    // visit_subset.shuffle(&mut rng);
    // let (route, score, length) = solve_random(&distance_matrix, &rewards, &visit_subset, false);
    // println!("Initial route: {:?}, Score: {}, Length: {}", route, score, length);
    // start_time = time::Instant::now();
    // let (optimized_route, optimized_score) = local_search(&mut route.clone(), &distance_matrix, &rewards, &mut score.clone(), NeighborhoodType::VertexSwap, false);
    // if start_time.elapsed().as_millis() > max_time as u128 {
    //     max_time = start_time.elapsed().as_millis() as i64;
    // }
    // println!("Vertex optimized Score: {}", optimized_score);
    // let (optimized_route, optimized_score) = local_search(&mut route.clone(), &distance_matrix, &rewards, &mut score.clone(), NeighborhoodType::EdgeSwap, false);
    // if start_time.elapsed().as_millis() > max_time as u128 {
    //     max_time = start_time.elapsed().as_millis() as i64;
    // }
    // println!("Edge optimized Score: {}", optimized_score);


     //run_search_tests(&distance_matrix, &rewards, 100);
    // run_candidate_tests(&distance_matrix, &rewards, 100);
    run_extended_tests(&distance_matrix, &rewards);



}
