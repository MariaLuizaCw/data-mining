import subprocess
import concurrent.futures

# Function to execute a Python file with arguments
def run_python_file(linha):
    try:
        print('Processing', linha)
        result = subprocess.run(['python', 'calculate_predictions_teste.py', str(linha)], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing process_bus_line.py for {linha}: {e}"

linhas = ['483', '864', '639', '309', '774', '629', 
				  '371', '397', '100', '838', '315', '624', '388', 
				  '918', '665', '328', '497', '878', '355', '138', '606', '457', '550', 
				  '803', '917', '638', '2336', '399', '298', '867', '553', '565', '422', 
				  '756', '292', '554', '634', '232', '415', '2803', '324', 
				  '852', '557', '759', '343', '779', '905', '108']

# Use ThreadPoolExecutor to run the scripts in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(run_python_file, linha) for linha in linhas]
    for future in concurrent.futures.as_completed(futures):
        try:
            output = future.result()
            print(output)
        except Exception as exc:
            print(f"Generated an exception: {exc}")