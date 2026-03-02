use std::env;
use std::error::Error;
use std::fs;

fn read_csv_mapped(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>>{
    let content = fs::read_to_string(file_path)?;
    let mut result : Vec<Vec<f64>> = Vec::new();
    for (_, line) in content.lines().enumerate(){
        if line.trim().is_empty() {
            continue;
        }
        let row: Result<Vec<f64>, _> = line.split(';').map(|value| value.trim().parse::<f64>()).collect();
        result.push(row?);
    }
    Ok(result)
}

fn convert_coordinates_to_length_matrix(coordinates: Vec<Vec<f64>>) -> Vec<Vec<i64>> {
    let num_points = coordinates.len();
    let mut length_matrix = vec![vec![0; num_points]; num_points];

    for i in 0..num_points {
        for j in 0..num_points {
            if i != j {
                let dx = coordinates[i][0] - coordinates[j][0];
                let dy = coordinates[i][1] - coordinates[j][1];
                length_matrix[i][j] = (((dx * dx + dy * dy).sqrt() + 0.5).floor()) as i64;
                length_matrix[j][i] = length_matrix[i][j];
            }
        }
    }
    length_matrix
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <csv_file_path>", args[0]);
        return;
    }
    let file_path = &args[1];
    let read_result = read_csv_mapped(file_path);
    let row_num = read_result.as_ref().map(|v| v.len()).unwrap_or(0);
    let col_num = read_result.as_ref().map(|v| v[0].len()).unwrap_or(0);
    let mut length_matrix: Vec<Vec<i64>> = Vec::new();
    //println!("{:?}", read_result); // Dla Pawła -> {:?} to debug, a {} to display. Print całego wektora/array działa tylko z {:?}
    if row_num == 0 || col_num == 0 {
        println!("The file is empty or not properly formatted.");
        return;
    }
    if row_num != col_num {
        println!("Assuming a set of coordinates, creating a length matrix");
        if col_num < 2 {
            println!("The file must contain at least two columns to create a length matrix.");
            return;
        }
        length_matrix = convert_coordinates_to_length_matrix(read_result.unwrap());
    }
    else {
        println!("Assuming a length matrix, using it directly");
        length_matrix = (read_result.unwrap()).into_iter().map(|row| row.into_iter().map(|value| value as i64).collect()).collect();
    }
    //println!("Length matrix: {:?}", length_matrix);
    //TODO - wagi jeszcze nie wiem jak interpretować
}
