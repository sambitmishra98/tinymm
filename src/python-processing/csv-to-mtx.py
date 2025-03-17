import os
import numpy as np
import scipy

def convert_csv_to_mtx(input_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                # Load CSV data
                try:
                    data = np.loadtxt(csv_path, delimiter=',')
                except Exception as e:
                    print(f"Error loading {csv_path}: {e}")
                    continue

                # Skip empty or malformed data
                if data.size == 0:
                    print(f"Skipping empty or malformed file: {csv_path}")
                    continue

                # Ensure data is 2D
                if data.ndim == 1:
                    data = data.reshape(1, -1)

                # Write dense Matrix Market file with fixed scientific notation
                dense_file = os.path.join(root, f'{base_name}-de.mtx')
                with open(dense_file, 'wb') as f_dense:
                    scipy.io.mmwrite(f_dense, data, comment='', field='real', precision=16)
                
                # Write sparse Matrix Market file with fixed scientific notation
                sparse_data = scipy.sparse.coo_matrix(data)
                sparse_file = os.path.join(root, f'{base_name}-sp.mtx')
                with open(sparse_file, 'wb') as f_sparse:
                    scipy.io.mmwrite(f_sparse, sparse_data, comment='', field='real', precision=16)

if __name__ == "__main__":
    input_directory = "../samples/pyfr/mats"
    convert_csv_to_mtx(input_directory)
