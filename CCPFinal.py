"""
This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see <https://www.gnu.org/licenses/>. 
"""


import pickle 
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector



###############################################
# Datasets generation
###############################################



# Returns a dataset.
# INPUTS
#   Index: index of the dataset type
#   randomSeed: Random seed used to generate dataset
# OUTPUTS
#   A (n,m) ndarray with n points with m coordinates in the range [0,1]
def get_dataset(index, randomSeed):
    
    np.random.seed(randomSeed)
    
    # QRAM
    if index == 1: 
        points= np.random.rand(4, 2)
        
    elif index == 2:
        points= np.random.rand(4, 3)
        
    elif index == 3:
        points= np.random.rand(4, 4)
    
    elif index == 4:
        points= np.random.rand(4, 5)    
    
    elif index == 5:
        points= np.random.rand(8, 2)
    
    elif index == 6:
        points= np.random.rand(8, 3)
        
    elif index == 7:
        points= np.random.rand(8, 4)
        
    elif index == 8:
        points= np.random.rand(8, 5)
        
    elif index == 9:
        points= np.random.rand(16, 2)
        
    elif index == 10:
        points= np.random.rand(16, 3)
        
    elif index == 11:
        points= np.random.rand(16, 4)
        
    elif index == 12:
        points= np.random.rand(16, 5)
        
    elif index == 13:
        points= np.random.rand(32, 2)

    elif index == 14: 
        points= np.random.rand(32, 3)
        
    elif index == 15: 
        points= np.random.rand(32, 4)
        
    elif index == 16: 
        points= np.random.rand(32, 5)
        
    # Amplitude embedding
    if index == 17: 
        points= np.random.rand(4, 2**2)
        
    elif index == 18:
        points= np.random.rand(4, 2**3)
        
    elif index == 19:
        points= np.random.rand(4, 2**4)
        
    elif index == 20:
        points= np.random.rand(4, 2**5)
        
    elif index == 21:
        points= np.random.rand(8, 2**2)
    
    elif index == 22:
        points= np.random.rand(8, 2**3)
        
    elif index == 23:
        points= np.random.rand(8, 2**4)
        
    elif index == 24:
        points= np.random.rand(8, 2**5)
        
    elif index == 25:
        points= np.random.rand(16, 2**2)
        
    elif index == 26:
        points= np.random.rand(16, 2**3)
        
    elif index == 27:
        points= np.random.rand(16, 2**4)
        
    elif index == 28:
        points= np.random.rand(16, 2**5)
        
    elif index == 29:
        points= np.random.rand(32, 2**2)

    elif index == 30: 
        points= np.random.rand(32, 2**3)
        
    elif index == 31: 
        points= np.random.rand(32, 2**4)
        
    elif index == 32: 
        points= np.random.rand(32, 2**5)

    # Encode the dataset
    if index < 17: # QRAM dataset
    
        # Normalize to [0,1]    
        min_v, max_v= np.min(points, axis=0), np.max(points, axis=0)
        points= (points-min_v)/(max_v-min_v)
        
        # Build |p_i>
        P= []
        for i in range( len(points) ):
        
            # Encode first coordinate of point points[i]
            p_i= [np.sqrt(points[i, 0]), np.sqrt(1 - points[i,0])]
            
            # Encode the remaining components as tensor product
            for k in range(1, points.shape[1]):
                p_i_k = [np.sqrt(points[i,k]), np.sqrt(1-points[i,k])]
                p_i = np.kron(p_i_k, p_i) # This way because of Qiskit's little endian
        
            assert(Statevector(p_i).is_valid())
        
            # Append |p_i> to P
            P.append(p_i)
        P= np.array(P)
        
    else: # Amplitude embedding with ||p_i||=1
        P= points/np.sqrt(np.sum(points**2, axis=1)).reshape(-1,1)
    
    return P



###############################################
# Classical brute force solution
###############################################


# Euclidean distance function
EuclideanDistance= lambda x,y : np.sqrt( np.sum( (x-y)**2 ) )


# Solves the Closest Pair of Points with a brute force method using Euclidean distance
# INPUTS
#   points: A (n,m) ndarray with n points with m coordinates in the range [0,1]
#   distance: callable to calculate the distance between two points
# OUTPUTS
#   a pair (i,j) with the row indices of the closest points in points
def BruteForce(points, distance):
    min_d= np.inf
    best_i, best_j= None, None
    mean_d= 0
    k= 0
    for i in range(len(points)-1):
        for j in range(i+1, len(points)):
            d= distance(points[i], points[j])
            mean_d+= d
            k+= 1
            if d < min_d:
                min_d, best_i, best_j= d, i, j
    return best_i, best_j, mean_d/k





###############################################
# Quantum algorithm solution
###############################################


# Implements the C-SWAPS operation in the article
# INPUTS
#   qc: Quantum circuit with registers 'd', 'rp', 'rpp'
#   m: The number of point coordinates
# OUTPUTS
#   A copy of qc with the C-SWAPS module at the end
def CSWAPS(qcSwaps: QuantumCircuit, m : int, d: QuantumRegister, r_p: QuantumRegister, r_pp: QuantumRegister):

    # Multi-qubit swap test
    qcSwaps.h(d)
    for i in range(m):
        qcSwaps.cswap(d, r_p[i], r_pp[i])
    qcSwaps.h(d)    



# Implements the encoding of P into |P> and |P'> in an input quantum circuit
# INPUTS
#   points: The points P to be encoded
#   d: Distance quantum register
#   r_i: Quantum register where indices of p_i are stored
#   r_p: Quantum register where p_i are stored
#   r_j: Quantum register where indices of x_j are stored
#   r_x: Quantum register where x_j are stored
# OUTPUTS
#   A copy of qc containing the prepared state |r_i>|r_p_i>|r_j>|r_p_j> 
def DataEncodingCircuit(points : np.ndarray, 
                 d : QuantumRegister,
                 r_i : QuantumRegister, r_j : QuantumRegister,
                 r_p : QuantumRegister, r_pp : QuantumRegister):

    # Set the number of points, size of |i>/|j>, and size of |p_i>/|p_j>
    n= len(points)
    logn= int(np.ceil(np.log2( n )))
    m= int(np.log2(points.shape[1]))
    
    # State representation of |P>
    P = np.zeros(2**(logn + m)) 

    # Encode the dataset
    for i in range( n ):
    
        # Encode the index of point points[i]
        current_i= np.zeros(2**logn)
        current_i[i] = 1
        
        # Encode coordinates in |p_i>
        p_i= points[i,:]
        
        # Create the state |i>|p_i>
        i_p_i= np.kron(p_i, current_i) # This way because of Qiskit's little endian
        
        # append |i>|p_i> to |P> as in Eq. 1 in the paper
        P+= i_p_i
        
    # Make norm equals to 1
    P /= np.sqrt( n )
    
    assert(Statevector(P).is_valid())
    
    # Create copy of P
    Pp= P.copy()
    
    qcEncoding= QuantumCircuit(d, r_i, r_p, r_j, r_x)
    
    # State preparation
    qcEncoding.initialize(P, r_i[:logn] + r_p[:m])
    qcEncoding.initialize(Pp, r_j[:logn] + r_pp[:m])
    return qcEncoding




# Merge |ij>+|ji> into |ij>
def MergeIJ(qc : QuantumCircuit, d: QuantumRegister, r_i : QuantumRegister, r_j: QuantumRegister):
    
    # Merge |ij>+|ji> into |ij>
    nqb= len(r_i)+len(r_j)+len(d)
    size= 2**(nqb)
    U= np.eye( size )
    S= 1/np.sqrt(2)
    for n_d in range(2):
        for i in range(0, 2**len(r_i)):
            for j in range(0, 2**len(r_i)):
                condition = (n_d == 0 and j<i)  or (n_d == 1 and j>i)
                if condition:
                    val1= bin(n_d)[2:] + bin(i)[2:].rjust(len(r_i), '0')[::-1] + bin(j)[2:].rjust(len(r_j), '0')[::-1] 
                    val2= bin(n_d)[2:] + bin(j)[2:].rjust(len(r_j), '0')[::-1] + bin(i)[2:].rjust(len(r_i), '0')[::-1]
                    val1= int( val1[::-1], 2)
                    val2= int( val2[::-1], 2)
                    U[val1,val1]= U[val1, val2]= U[val2, val1]= S
                    U[val2,val2]= -S
                    
    Ug= UnitaryGate( U )

    qc.append(Ug, [d[0]]+r_i[:] + r_j[:])



# Create the quantum circuit of Fig. 2 in the paper
# INPUT
#   d: Distance quantum register
#   r_i: Quantum register where indices of p_i are stored
#   r_p: Quantum register where p_i are stored
#   r_j: Quantum register where indices of x_j are stored
#   r_x: Quantum register where x_j are stored
# OUTPUT
#   A quantum circuit analog to Fig. 2 to find the closest pair of points of P
def CreateProcessingCircuit(d : QuantumRegister,
                            r_i : QuantumRegister, r_p : QuantumRegister, 
                            r_j : QuantumRegister, r_x : QuantumRegister):
    
    qc= QuantumCircuit(d, r_i, r_p, r_j, r_x)
    
    # SWAP test
    CSWAPS(qc, m, d, r_p, r_x)
    
    # Merge |i>|j> + |j>|i>
    MergeIJ(qc, d, r_i, r_j)
    MergeIJ(qc, d, r_p, r_x)
    
    return qc


# Calculate the scores of an experiment
# INPUT
#   n: The number of points in the set
#   counts: The counts provided by measurement
# OUTPUT
#   A list of tuples (i, j, score) sorted in ascending order by score.
#       i: Index of a point in P
#       j: Index of a point in P'
#       score: The score of the i-th point in P and the j-th point in P', i<j
def getScoresEuclidean(n, counts):
    
    # Create a table to count #|1>'s and #|0>'s for each pair of measured points i,j
    table= {}
    for count in counts:
        d= int(count[-1], 2) # Because of Qiskit's little endian
        pair= count[:-1] # Pair of points
        
        bin_i= pair[:(len(pair)//2)] # Index |i> in binary
        bin_j= pair[(len(pair)//2):] # Index |j> in binary
        i= int(bin_i, 2)
        j= int(bin_j, 2)
        
        # Discard point if i==j
        if i!=j:
            if i>j:
                i,j= j,i
            if i not in table:
                table[i]= {}
            if j not in table[i]:
                table[i][j]= [0,0]
            table[i][j][d]+= counts[count]

    # Create list of tuples (i, j, score)
    L= []
    for i in table:
        for j in table[i]:
            n0s, n1s= table[i][j][0], table[i][j][1]
            inner= 1-2*n1s/(n1s+n0s)
            if inner < 0:
                inner= 0
            score_ij= np.sqrt(2-2*np.sqrt(inner))
            L.append( (i, j, score_ij) )

    # Sort tuples by score
    L= sorted(L, key= lambda x: x[2])
    return L


# Auxiliary method to speed up measurement simulation and
# to use numpy for measurement (improves replicability of results)
def measure(sv : Statevector, qubits : list[int], shots:int):
    
    svdict= sv.probabilities_dict() # Dictionary of probabilities
    
    probs= {}
    for outcome in svdict:
        rev_outcome= outcome[::-1]
        ket= ''.join([rev_outcome[i] for i in qubits])
        p_val= svdict[outcome]
        if not np.isclose(p_val, 0):
            ket= ket[::-1]
            if ket not in probs:
                probs[ket]= 0.0
            probs[ket]+= p_val
    
    
    ket_list= []
    prob_list= []
    for ket in probs:
        ket_list.append(ket)
        prob_list.append(probs[ket])
    ket_list= np.array(ket_list, copy=False)
    
    prob_list= np.array(prob_list, copy=False)/np.sum(prob_list)
    
    selected= np.random.choice(a= ket_list, size=shots, p=prob_list)
    selected= np.sort(selected)

    counts= {}
    for ket in selected:
        if ket not in counts:
            counts[ket]= 1
        else:
            counts[ket]+= 1
    return counts
    


###############################################
# Experimentation
###############################################
file= 'FinalResults.pkl'
exper= 200
shots= {1 :50000, 2 :50000, 3 :50000, 4 :50000, 
        5 :200000, 6 :200000, 7 :200000, 8: 200000,
        9 :2000000, 10: 2000000, 11: 2000000, 12:2000000,
        13:10000000, 14:10000000, 15:10000000, 16:10000000}
size= len(shots)
for i in range(1, size+1):
    shots[i+size]= shots[i]
initialSeed= 1234
currentSeed= initialSeed
# Load previous results
try:
    with open(file, 'rb') as f:
        results = pickle.load(f)
        print('Results (success in experiments):')
        for current_dataset in results:
            print('Dataset {}: {}/{}'.format(current_dataset, results[current_dataset]['success'], results[current_dataset]['performed']))
            
            points= get_dataset(current_dataset, currentSeed)
            dim= int(np.log2(points.shape[1]) if current_dataset < 5 else points.shape[1])
            m= int(np.log2(points.shape[1]))
            logn= int(np.ceil(np.log2( len(points) )))
            n_shots= 1+2*logn
            print('\t{} points of dimension {}. Qubits: {}. Measurements: {}'.format(points.shape[0], dim, 1+2*(logn+m), shots[current_dataset]), flush=True)
        
except:
    print('No hay resultados')
    results= {}

    
# Simulator instance
sim= AerSimulator()


print('\n\nSTART SIMULATION\n\n')
execution_order= []
for i in range(1, len(shots)//2+1): # First execute the easiest dataset types
    execution_order.append(i)
    execution_order.append(i+len(shots)//2)
for current_dataset in execution_order:

    if current_dataset not in results: # Dataset already processed
        results[current_dataset]= {'success' : 0, 'performed' : 0}
    else:
        maxExper= results[current_dataset]['performed']
        if maxExper == exper:
            currentSeed+= exper
            continue


    # Get current dataset
    points= get_dataset(current_dataset, currentSeed)    
    n= len(points) # Number of points in the dataset
    logn= int(np.ceil(np.log2( n )))
    m= int(np.log2(points.shape[1])) # Number of coordinates in the dataset
    dim= int(np.log2(points.shape[1]) if current_dataset <= len(shots)//2 else points.shape[1]) # Points dimension

    print('Dataset selected: {}'.format(current_dataset), flush=True)
    print('\t{} points of dimension {}'.format(points.shape[0], dim), flush=True)
    

    # Create main circuit
    print('Generating solution circuit...', flush=True)
    d= QuantumRegister(1, 'd') # Store the distance
    r_i= QuantumRegister(logn, 'ri') # Stores the indices of points in the set P
    r_p= QuantumRegister(m, 'rp') # Stores the set P
    r_j= QuantumRegister(logn, 'rj') # Stores the indices of points in the set P'
    r_x= QuantumRegister(m, 'rx') # Stores the set X

    # Create the circuit    
    
    
    # Processing circuit
    qcProcessing= CreateProcessingCircuit(d, r_i, r_p, r_j, r_x)
    print('\tRequired number of qubits: {}'.format(qcProcessing.num_qubits))
    
    # Prepare simulation
    n_shots= shots[current_dataset] #2048*(2**(1+2*logn)) # At least 1024 samples per possible output
    
    # Prepare measurement
    qubits= [0]+list(range(1, logn+1))+list(range(1+logn+m, 1+2*logn+m))

    
        
        
    for current_experiment in range(exper):

        if current_experiment < results[current_dataset]['performed']:
            currentSeed+= 1
            continue
        

        # Get the data
        points= get_dataset(current_dataset, currentSeed)
        currentSeed+= 1
        print('\nDataset {}. Experiment {}/{}.'.format(current_dataset, current_experiment+1, exper), flush= True)
        
        
        
        # Brute force search
        print('Finding closest points with Brute Force...', flush=True)
        BFp_i, BFp_j, meand_BF= BruteForce(points, EuclideanDistance)
        d_BF= EuclideanDistance(points[BFp_i], points[BFp_j])
        
        # Quantum proposal
        
        
        # Data encoding circuit
        qcEncoding= DataEncodingCircuit(points, d, r_i, r_j, r_p, r_x)
        
        # Compose with main circuit
        qc= qcEncoding.compose(qcProcessing)
        qc.save_statevector() # Save statevector for measurement (faster than using measure())

        # Simulation
        qct= transpile(qc, sim)
        sv= sim.run(qct, shots= 1).result().get_statevector(qct)

        print('Starting simulation with {} shots...'.format(n_shots), flush=True)
        
        # Sample measurement |d>|i>|j>
        #counts= sv.sample_counts(n_shots, qubits)
        counts= measure(sv, qubits, n_shots)
        
        
        print('Computing scores...', flush=True)
        scores= getScoresEuclidean(len(points), counts)
        Qp_i, Qp_j, QEuc= scores[0]
        d_Q= EuclideanDistance(points[Qp_i], points[Qp_j])
            
        success= ((BFp_i == Qp_i) and (BFp_j == Qp_j)) or (d_BF == d_Q)
        
        print('\nEnd of simulation. SUCCESS: {}'.format(success))
        
        
        print('Brute force search: distance of {}'.format(EuclideanDistance( points[BFp_i], points[BFp_j] )))
        print('\t Points {}-{}'.format(BFp_i, BFp_j))
        
        print('Quantum search: distance of {}'.format(QEuc))
        print('\t Points {}-{}'.format(Qp_i, Qp_j))
        
        if not success:
            true_dQ= None
            for i, j, dQ in scores:
                if i==BFp_i and j==BFp_j:
                    true_dQ= dQ
            print('\tGround truth points {}-{} with distance of {}'.format(BFp_i, BFp_j, true_dQ))

        
            
        results[current_dataset]['performed']+= 1
        if current_experiment == 0:
            results[current_dataset]['true distance']= []
            results[current_dataset]['calculated distance']= []
            results[current_dataset]['mean true distance']= []
        results[current_dataset]['true distance'].append(d_BF)
        results[current_dataset]['mean true distance'].append(meand_BF)
        results[current_dataset]['calculated distance'].append(QEuc)
        
        if success:
            results[current_dataset]['success']+= 1
        print('\tSuccess for now: {}/{}'.format(results[current_dataset]['success'], results[current_dataset]['performed']))
        print('\n')

        # Save results
        with open(file, 'wb') as f:
            pickle.dump(results, f)


        
# Load and print results
with open(file, 'rb') as f:
    results = pickle.load(f)
    
print('Results (success in experiments):')
for current_dataset in results:
    print('Dataset {}: {}/{}'.format(current_dataset, results[current_dataset]['success'], results[current_dataset]['performed']))
    
    points= get_dataset(current_dataset, currentSeed)
    dim= int(np.log2(points.shape[1]) if current_dataset < 5 else points.shape[1])
    m= int(np.log2(points.shape[1]))
    logn= int(np.ceil(np.log2( len(points) )))
    n_shots= 1+2*logn
    print('\t{} points of dimension {}. Qubits: {}. Measurements: {}'.format(points.shape[0], dim, 1+2*(logn+m), shots[current_dataset]), flush=True)

    
