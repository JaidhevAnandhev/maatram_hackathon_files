<?php
// Enable error reporting for debugging
error_reporting(E_ALL);
ini_set('display_errors', 1);

// Set the content type to JSON
header('Content-Type: application/json');

// Allow CORS
header('Access-Control-Allow-Origin: *'); // Allow all origins
header('Access-Control-Allow-Methods: POST, GET, OPTIONS'); // Allow specific methods
header('Access-Control-Allow-Headers: Content-Type'); // Allow specific headers

// Establish a database connection
$servername = "localhost:3307"; // Adjust the port if necessary
$username = "root";
$password = "";
$dbname = "maatram_db";

$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die(json_encode(['error' => 'Connection failed: ' . $conn->connect_error]));
}


header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit; // Exit for preflight requests
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = isset($_POST['name']) ? trim($_POST['name']) : '';
    $response = isset($_POST['response']) ? trim($_POST['response']) : '';
    $pattern_score = isset($_POST['pattern_score']) ? intval($_POST['pattern_score']) : null;

    if (empty($name) || empty($response) || $pattern_score === null) {
        echo json_encode(['error' => 'Please fill in all fields.']);
        exit;
    }

    $stmt = $conn->prepare("INSERT INTO verifications (name, response, pattern_score) VALUES (?, ?, ?)");
    $stmt->bind_param("ssi", $name, $response, $pattern_score);

    if ($stmt->execute()) {
        echo json_encode(['message' => "New record created successfully", 'pattern_score' => $pattern_score]);
    } else {
        echo json_encode(['error' => 'Failed to create record.']);
    }
    $stmt->close();
} elseif ($_SERVER['REQUEST_METHOD'] === 'GET') {
    $result = $conn->query("SELECT * FROM verifications");

    if ($result->num_rows > 0) {
        $data = [];
        while ($row = $result->fetch_assoc()) {
            $data[] = $row;
        }
        echo json_encode($data);
    } else {
        echo json_encode(['message' => 'No records found.']);
    }
}

$conn->close();

?>