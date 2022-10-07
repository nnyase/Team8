import pickle

# Store results in a pkl file
def store_in_pkl(pathStore, results):
    # Open the file result.pkl and create it if it doesn't exist
    with open(pathStore + 'result.pkl', 'wb') as f:
        pickle.dump(results, f)

# Read the pkl file 
def read_pkl(sPath):
    with open(sPath, 'rb') as f:
        result = pickle.load(f)
    
    return result