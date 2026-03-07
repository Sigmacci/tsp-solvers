use std::env;
use std::error::Error;
use std::fs;
use rand::Rng;
use rand::seq::IteratorRandom;

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

fn solve_random(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<u64>) -> (Vec<u64>, i64) {
    let mut _visit_subset : Vec<u64> = visit_subset.clone();
    let mut total_score   : i64      = 0;

    for i in 0.._visit_subset.len() {
        let from = _visit_subset[i       % _visit_subset.len()] as usize;
        let to   = _visit_subset[(i + 1) % _visit_subset.len()] as usize;

        total_score += rewards[to] - distance_matrix[from][to];
    }

    (_visit_subset, total_score)
}

fn solve_2_regret(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<u64>) -> (Vec<u64>, i64) {
    solve_2_regret_weighted(distance_matrix, rewards, visit_subset, 0.0)
}

fn solve_2_regret_weighted(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>, visit_subset: &Vec<u64>, weight: f64) -> (Vec<u64>, i64) {
    let mut _visit_subset : Vec<u64> = visit_subset.clone();
    let mut total_score   : i64      = 0;
    let mut route         : Vec<u64> = Vec::new();

    for _ in 0..2 {
        route.push(_visit_subset.remove(0));
    }

    for i in 0..route.len() {
        let u = route[i] as usize;
        let v = route[(i + 1) % route.len()] as usize;
        total_score += rewards[u] - distance_matrix[u][v];
    }

    while !_visit_subset.is_empty() {
        let mut best_regret  : i64   = i64::MIN;
        let mut best_point   : u64   = 0;
        let mut best_cost    : i64   = 0;
        let mut best_postion : usize = 0;

        for &point in &_visit_subset {
            let mut best_insertion_cost    : i64   = i64::MAX;
            let mut second_best_cost       : i64   = i64::MAX;
            let mut insertion_index        : usize = 0;

            for i in 0..route.len() {
                let from : usize = route[i       % route.len()] as usize;
                let to   : usize = route[(i + 1) % route.len()] as usize;

                let insertion_cost = distance_matrix[from][point as usize] + distance_matrix[point as usize][to] - distance_matrix[from][to];

                if insertion_cost < best_insertion_cost {
                    second_best_cost       = best_insertion_cost;
                    best_insertion_cost    = insertion_cost;
                    insertion_index        = i + 1;
                } else if insertion_cost < second_best_cost {
                    second_best_cost       = insertion_cost;
                }
            }

            let regret = second_best_cost - best_insertion_cost - (weight * best_insertion_cost as f64) as i64;
            if regret > best_regret {
                best_regret = regret;
                best_point  = point;
                best_cost   = best_insertion_cost;
                best_postion = insertion_index;
            }
        }
        route.insert(best_postion, best_point);
        total_score += rewards[best_point as usize] - best_cost;
        _visit_subset.retain(|&x| x != best_point);
    }

    (route, total_score)
}

fn calculate_score(route: &Vec<u64>, distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> i64 {
    let mut total_score: i64 = 0;

    for i in 0..(route.len() - 1) {
        let from = route[i    ] as usize;
        let to   = route[i + 1] as usize;

        total_score += rewards[to] - distance_matrix[from][to];
    }
    total_score += rewards[route[0] as usize] - distance_matrix[route[route.len() - 1] as usize][route[0] as usize];

    total_score
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
    //println!("{:?}", read_result); // Dla Pawła -> {:?} to debug, a {} to display. Print całego wektora/array działa tylko z {:?}
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

    let mut rng = rand::thread_rng();

    let     num_points   : u64      = rng.gen_range(2..distance_matrix.len() as u64); 
    let     visit_subset : Vec<u64> = (0..distance_matrix.len() as u64).choose_multiple(&mut rng, num_points as usize);

    // println!("Distance matrix: {:?}", rewards);
    let (random_solution, random_score) = solve_random(&distance_matrix, &rewards, &visit_subset);
    let (regret_solution, regret_score) = solve_2_regret(&distance_matrix, &rewards, &visit_subset);
    let (regret_weighted_solution, regret_weighted_score) = solve_2_regret_weighted(&distance_matrix, &rewards, &visit_subset, -1.5);

    let calculated_score = calculate_score(&random_solution, &distance_matrix, &rewards);
    let calculated_score_regret = calculate_score(&regret_solution, &distance_matrix, &rewards);
    let calculated_score_regret_weighted = calculate_score(&regret_weighted_solution, &distance_matrix, &rewards);

    assert_eq!(random_score, calculated_score, "The calculated score does not match the expected score.");
    assert_eq!(regret_score, calculated_score_regret, "The calculated score for the regret solution does not match the expected score.");
    assert_eq!(regret_weighted_score, calculated_score_regret_weighted, "The calculated score for the weighted regret solution does not match the expected score.");

    println!("Random solution: {:?} with score: {}", random_solution, random_score);
    println!("Regret solution: {:?} with score: {}", regret_solution, regret_score);
    println!("Regret weighted solution: {:?} with score: {}", regret_weighted_solution, regret_weighted_score);
    //TODO - wagi jeszcze nie wiem jak interpretować
}
