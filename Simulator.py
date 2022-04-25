'''
Simulator.py

Class of Simulator

'''
import numpy as np
import random

class Simulator():

    def __init__(self):
        self.size_i = None
        self.size_g = None
        self.DS = []
        self.Desired_Counts = None  # Desired Q(j)
        self.Current_Sampled_Count = None  # Current O(j)
        self.pj = None  # Frequency of all j

    def __repr__(self):
        string = "Desired Counts:" + self.Desired_Counts.__repr__() + '\n'
        string +=" Current_Sampled_Count:" + self.Current_Sampled_Count.__repr__() + '\n'
        for i in range(self.size_i):
            string += "[Dataset" + str(i) + "]: " + self.DS[i].__repr__() + '\n'
        return string

    def Scenario_SimilarDataSet_Equal_Distribution(self):
        DG_distribution = np.array([1/3,1/3,1/3])
        self.Desired_Counts = np.array([1000,1000,1000],dtype=int)
        self.Current_Sampled_Count = np.zeros_like(self.Desired_Counts)
        Ns  = np.random.normal(10000,10000*0.05,20)
        for i in range(20):
            self.DS.append(DataSet(int(Ns[i]),DG_distribution))
        self.size_i = len(self.DS)
        self.size_g = len(self.Desired_Counts)

    def Scenario_SimilarDataSet_Skewed_Distribution(self):
        DG_distribution = np.array([0.2,0.3,0.5])
        self.Desired_Counts = np.array([1000,1000,1000],dtype=int)
        self.Current_Sampled_Count = np.zeros_like(self.Desired_Counts)
        Ns  = np.random.normal(10000,10000*0.05,20)
        for i in range(20):
            self.DS.append(DataSet(int(Ns[i]),DG_distribution))
        self.size_i = len(self.DS)
        self.size_g = len(self.Desired_Counts)

    def Scenario_SimilarDataSet_Very_Skewed_Distribution(self):
        DG_distribution = np.array([0.1,0.2,0.7])
        self.Desired_Counts = np.array([1000,1000,1000],dtype=int)
        self.Current_Sampled_Count = np.zeros_like(self.Desired_Counts)
        Ns  = np.random.normal(10000,10000*0.05,20)
        for i in range(20):
            self.DS.append(DataSet(int(Ns[i]),DG_distribution))
        self.size_i = len(self.DS)
        self.size_g = len(self.Desired_Counts)

    def Scenario_SkewedDataSet_Very_Skewed_Distribution(self):
        DG_distribution = np.array([0.1,0.2,0.7])
        self.Desired_Counts = np.array([1000,1000,1000],dtype=int)
        self.Current_Sampled_Count = np.zeros_like(self.Desired_Counts)
        Ns  = np.random.normal(10000,10000*0.05,20)
        Ns[5] = 30000
        Ns[6] = 1000
        for i in range(20):
            self.DS.append(DataSet(int(Ns[i]),DG_distribution))
        self.size_i = len(self.DS)
        self.size_g = len(self.Desired_Counts)


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
            if np.random.uniform(0,1) < unused_count / total_count: # The prob for this chosen sample to be a new sample is unused/total
                result[random_choice] += 1
                Dataset.DG_unused_count[random_choice] -= 1
        for j in range(self.size_g):
            if self.Current_Sampled_Count[j] + result[j] > self.Desired_Counts[j]:
                result[j] = self.Desired_Counts[j] - self.Current_Sampled_Count[j]
                self.Current_Sampled_Count[j] =self.Desired_Counts[j]
            else:
                self.Current_Sampled_Count[j] += result[j]
        return result

    def check_complete(self):
        for i in range(self.size_g):
            if self.Desired_Counts[i] > self.Current_Sampled_Count[i]:
                return False
        return True


    def get_frequency(self,j):
        if self.pj is None:
            self.pj = np.zeros(self.size_g)
            for j_prime in range(self.size_g):
                sum_N = 0
                sum_Nj = 0
                for dataset in self.DS:
                    sum_N += dataset.N
                    sum_Nj += dataset.DG_total_count[j_prime]
                self.pj[j_prime] = sum_Nj/sum_N
            return self.pj[j]
        else:
            return self.pj[j]

class DataSet():


    def __init__(self,N,distribution):
        '''
        Initalize a dataset with sample count of N, distribution specified.
        Note that here introduces some noise to the distribution
        :param N: Total number of sampels within the dataset.
        :param distribution: Distribuition requirement(Need to have sum of 1)
        '''
        self.size_g = None
        self.N = None  # Total N_i
        self.DG_total_count = None  # Total N_i(j)
        self.DG_unused_count = None  # Total N_i(j) - O_i(j)
        self.c0 = None
        self.cx = None

        self.size_g = len(distribution)
        self.N = N

        self.DG_total_count = np.zeros(self.size_g,dtype=int)
        for i in range(self.size_g-1):
            noise = np.random.uniform(-0.3, 0.3)
            self.DG_total_count[i] = int(N*distribution[i]*(1+noise)) # Estimate the count with 10% noise
        self.DG_total_count[-1] = N-sum(self.DG_total_count) # Make sure eventually the size is N

        # Copy the unused count
        self.DG_unused_count = self.DG_total_count.copy()

        # Set costs
        self.c0 = N
        self.cx = 0  #fix cx for all dataset TODO: Discuss if this is feasible


    @staticmethod
    def sample_for_distribution(distribution):
        thre = np.random.uniform(0, np.sum(distribution))
        sum = 0
        for i in range(len(distribution)):
            sum += distribution[i]
            if sum >= thre:
                return i

    # Get O_i
    def get_total_sampled_count(self):
        return np.sum(self.DG_total_count) - np.sum(self.DG_unused_count)


    def __repr__(self):
        if self.DG_total_count is None or self.DG_unused_count is None or self.size_g is None:
            return "Not Initialized"

        string = ""
        for i in range(self.size_g-1):
            string += str(self.DG_unused_count[i]) + '/' +  str(self.DG_total_count[i])  + ', '
        string += str(self.DG_unused_count[self.size_g-1]) + '/' + str(self.DG_total_count[self.size_g-1])
        return 'N='+str(self.N)+' [' + string +']'

    def get_cost(self,k):
        return self.c0 + k * self.cx


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    sim = Simulator()

    print(sim.sample(0,1))
    pass