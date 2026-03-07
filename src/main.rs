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

fn get_random_solution(distance_matrix: &Vec<Vec<i64>>, rewards: &Vec<i64>) -> (Vec<u64>, i64) {
    let mut rng = rand::thread_rng();

    let     num_points   : u64      = rng.gen_range(2..distance_matrix.len() as u64); // In the context of TSP, we need at least 2 points to create a valid route
    let     visit_subset : Vec<u64> = (0..distance_matrix.len() as u64).choose_multiple(&mut rng, num_points as usize);
    let mut total_score  : i64      = 0;

    for i in 0..(visit_subset.len() - 1) {
        let from = visit_subset[i    ] as usize;
        let to   = visit_subset[i + 1] as usize;

        total_score += rewards[to] - distance_matrix[from][to];
    }
    total_score += rewards[visit_subset[0] as usize] - distance_matrix[visit_subset[visit_subset.len() - 1] as usize][visit_subset[0] as usize];

    (visit_subset, total_score)
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

    // println!("Distance matrix: {:?}", rewards);
    let (solution, score) = get_random_solution(&distance_matrix, &rewards);
    // let calculated_score = calculate_score(&solution, &distance_matrix, &rewards);
    // assert_eq!(score, calculated_score, "The calculated score does not match the expected score.");
    println!("Random solution: {:?} with score: {}", solution, score);
    //TODO - wagi jeszcze nie wiem jak interpretować
}
