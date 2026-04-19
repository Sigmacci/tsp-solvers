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

        let mut dump_filename : String = "./solutions/solution_".to_owned() + methods[idx] + "_a_.csv";
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

fn local_search(route: &mut Vec<usize>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, score: &mut i64, neighborhood_type: NeighborhoodType, greedy: bool) -> (Vec<usize>, i64) {
    let mut rng = rand::thread_rng();
    let mut improved: bool = true;
    let mut last_score = *score;
    while improved {
        //println!("I am actually working, route: {:?}, score: {}, last_score: {}", route, score, last_score);
        let neighborhood = initialize_neighborhood(route, distance_matrix, rewards, neighborhood_type);
        improved = false;

        let best_option = if greedy {
            neighborhood.iter()
                .filter(|&(_, &delta_score)| delta_score > 0)
                .choose(&mut rand::thread_rng())
        } else {
            neighborhood.iter().filter(|&(_, &delta_score)| delta_score > 0).max_by_key(|&(_, delta_score)| *delta_score)
        };

        if let Some((&best_move, &delta_score)) = best_option {
            if delta_score > 0 {
                *score += delta_score;
                last_score = *score;
                improved = true;
                match best_move {
                    Move::ExtInsert { node, after_idx } => {
                        route.insert(after_idx , node);
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
                    + "_a_.csv";

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
                    let (optimized_route, optimized_score) = local_search(
                        &mut route.clone(),
                        &distance_matrix,
                        &rewards,
                        &mut score.clone(),
                        *neighborhood_type,
                        greedy,
                    );
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
                + "_a_.csv";

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
    let k = 10;
    let n = distance_matrix.len();
    let mut candidate_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut dists: Vec<(usize, i64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, distance_matrix[i][j]))
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        candidate_neighbors[i] = dists.iter().take(k).map(|&(j, _)| j).collect();
    }
    let mut best_route = route.clone();
    let mut best_score = calculate_score(route, distance_matrix, rewards);
    let mut improved = false;

    let mut best_move: Option<(Move, i64)> = None;
    let mut best_move_type = NeighborhoodType::EdgeSwap;

    for &neigh_type in &[NeighborhoodType::EdgeSwap, NeighborhoodType::VertexSwap] {
        let neighborhood = initialize_neighborhood(route, distance_matrix, rewards, neigh_type);
        for (mv, &delta) in neighborhood.iter() {
            if delta <= 0 { continue; }
            let mut introduces_candidate = false;
            match mv {
                Move::IntEdgeSwap { reverse_start, reverse_end } => {
                    let n = route.len();
                    let i = *reverse_start;
                    let j = *reverse_end;
                    let a = route[(i + n - 1) % n];
                    let b = route[j];
                    let c = route[i];
                    let d = route[(j + 1) % n];
                    if candidate_neighbors[a].contains(&b) || candidate_neighbors[b].contains(&a)
                        || candidate_neighbors[c].contains(&d) || candidate_neighbors[d].contains(&c) {
                        introduces_candidate = true;
                    }
                },
                Move::IntVertexSwap { idx1, idx2 } => {
                    let n = route.len();
                    let i = *idx1;
                    let j = *idx2;
                    let prev1 = route[(i + n - 1) % n];
                    let node1 = route[i];
                    let next1 = route[(i + 1) % n];
                    let prev2 = route[(j + n - 1) % n];
                    let node2 = route[j];
                    let next2 = route[(j + 1) % n];
                    let edges = [ (prev1, node2), (node2, next1), (prev2, node1), (node1, next2) ];
                    for &(u, v) in &edges {
                        if candidate_neighbors[u].contains(&v) || candidate_neighbors[v].contains(&u) {
                            introduces_candidate = true;
                            break;
                        }
                    }
                },
                Move::ExtInsert { node, after_idx } => {
                    let n = route.len();
                    let prev = route[*after_idx % n];
                    let next = route[(*after_idx + 1) % n];
                    if candidate_neighbors[prev].contains(node) || candidate_neighbors[*node].contains(&prev)
                        || candidate_neighbors[*node].contains(&next) || candidate_neighbors[next].contains(node) {
                        introduces_candidate = true;
                    }
                },
                Move::ExtRemove { .. } => {
                    //musze dodać, mimo że nie może być. Niesamowite
                },
            }
            if introduces_candidate {
                let new_score = best_score + delta;
                if new_score > best_score {
                    best_score = new_score;
                    best_move = Some((*mv, delta));
                    best_move_type = neigh_type;
                    improved = true;
                }
            }
        }
    }

    if let Some((mv, delta)) = best_move {
        let mut new_route = route.clone();
        match mv {
            Move::ExtInsert { node, after_idx } => {
                new_route.insert(after_idx, node);
            },
            Move::ExtRemove { idx } => {
                new_route.remove(idx);
            },
            Move::IntVertexSwap { idx1, idx2 } => {
                new_route.swap(idx1, idx2);
            },
            Move::IntEdgeSwap { reverse_start, reverse_end } => {
                new_route[reverse_start..=reverse_end].reverse();
            }
        }
        let score = calculate_score(&new_route, distance_matrix, rewards);
        return (new_route, score);
    }
    (route.clone(), best_score)
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

        let filename = format!("./solutions/solution_{}_a_.csv", method_name);
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
            let elapsed = start_time.elapsed().as_millis() as i64;
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
    // let (route, score, length) = solve_2_regret(&distance_matrix, &rewards, &visit_subset, false);
    // println!("Initial route: {:?}, Score: {}, Length: {}", route, score, length);
    // start_time = time::Instant::now();
    // let (optimized_route, optimized_score) = local_search(&mut route.clone(), &distance_matrix, &rewards, &mut score.clone(), false, NeighborhoodType::VertexSwap);
    // if start_time.elapsed().as_millis() > max_time as u128 {
    //     max_time = start_time.elapsed().as_millis() as i64;
    // }
    // println!("Optimized route: {:?}, Score: {}", optimized_route, optimized_score);

    run_search_tests(&distance_matrix, &rewards, 100);
    //run_candidate_tests(&distance_matrix, &rewards, 100);

}
