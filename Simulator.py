'''
Simulator.py

Class of Simulator

'''
import numpy as np
import random

class Simulator():
    size_i = None
    size_g = None
    DS = []
    Desired_Counts = None #Desired Q(j)
    Current_Sampled_Count= None #Current O(j)

    def __init__(self,type=1):
        if type==1:
            self.create_DS_type1()
        else:
            raise NotImplementedError
        self.size_i = len(self.DS)
        self.size_g = len(self.Desired_Counts)

    #TODO: Need to find more definition for creating different types of dataests
    def create_DS_type1(self):
        '''
          Creating 10 DataSets(i); 3 Demographic Groups(j)
          Total Count N=200~5000
          C0 = N
          Cx fixed = 1
        '''
        DG_distribution = np.array([0.1,0.3,0.6])
        self.Desired_Counts = np.array([30,30,40],dtype=int)
        self.Current_Sampled_Count = np.zeros_like(self.Desired_Counts)
        for _ in range(10):
            N = np.random.randint(100,1000)
            self.DS.append(DataSet(N,DG_distribution))

    def sample(self,i,k=1):
        '''
        Takes k query from the dataset i.
        Note: This simulation does NOT assume the query returns k different results(that is, it is query repeated for k times)
        :param i: Index of dataset
        :param k: number of samples to query
        :return: Array of valid(new) samples per demographic group
        '''
        Dataset = self.DS[i]
        result = np.zeros(Dataset.size_g, dtype=int)
        for try_index in range(k):
            random_choice = DataSet.sample_for_distribution(Dataset.DG_total_count)
            if self.Current_Sampled_Count[random_choice] >= self.Desired_Counts[random_choice]: # If this entry is already full, skip it
                continue
            unused_count = Dataset.DG_unused_count[random_choice]
            total_count = Dataset.DG_total_count[random_choice]
            if np.random.uniform(0,1) < unused_count / total_count:  # The prob for this chosen sample to be a new sample is unused/total
                result[random_choice] += 1
                Dataset.DG_unused_count[random_choice] -= 1
        self.Current_Sampled_Count += result # Note: at k>1, this may overflow(w.r.t to desired count)
        return result

    def check_complete(self):
        for i in range(self.size_g):
            if self.Desired_Counts[i] > self.Current_Sampled_Count[i]:
                return False
        return True

class DataSet():
    size_g = None
    N = None            # Total N_i
    DG_total_count=None # Total N_i(j)
    DG_unused_count=None # Total O_i(j)
    c0 = None
    cx = None

    def __init__(self,N,distribution):
        '''
        Initalize a dataset with sample count of N, distribution specified.
        Note that here introduces some noise to the distribution
        :param N: Total number of sampels within the dataset.
        :param distribution: Distribuition requirement(Need to have sum of 1)
        '''
        self.size_g = len(distribution)
        self.N = N

        self.DG_total_count = np.zeros(self.size_g,dtype=int)
        for i in range(self.size_g-1):
            noise = np.random.uniform(-0.1, 0.1)
            self.DG_total_count[i] = int(N*distribution[i]*(1+noise)) # Estimate the count with 10% noise
        self.DG_total_count[-1] = N-sum(self.DG_total_count) # Make sure eventually the size is N

        # Copy the unused count
        self.DG_unused_count = self.DG_total_count.copy()

        # Set costs
        self.c0 = N
        self.cx = 0.1 #fix cx for all dataset TODO: Discuss if this is feasible


    @staticmethod
    def sample_for_distribution(distribution):
        thre = np.random.uniform(0, np.sum(distribution))
        sum = 0
        for i in range(len(distribution)):
            sum += distribution[i]
            if sum >= thre:
                return i

    def __repr__(self):
        if self.DG_total_count is None or self.DG_unused_count is None or self.size_g is None:
            return "Not Initialized"

        string = ""
        for i in range(self.size_g-1):
            string += str(self.DG_unused_count[i]) + '/' +  str(self.DG_total_count[i])  + ', '
        string += str(self.DG_unused_count[self.size_g-1]) + '/' + str(self.DG_total_count[self.size_g-1])
        return 'N='+str(self.N)+' [' + string +']'


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    sim = Simulator()

    print(sim.sample(0,1))
    pass