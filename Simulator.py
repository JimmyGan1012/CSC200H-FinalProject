'''
Simulator.py

Class of Simulator

'''
import numpy as np

class Simulator():

    def __init__(self,numDS,numDG):
        #TODO: Need to find a definition for creating different types of dataests
        pass

    def create_DS(self):
        pass


class DataSet():
    def __init__(self,DG_counts):
        self.DG_total_count = DG_counts.copy()
        # Define a noise of 10% on each DG count
        noise = np.random.uniform(-0.1, 0.1)
        for i in range(len(self.DG_total_count)):
            self.DG_total_count[i] = int(self.DG_total_count[i] *(1+noise))
        self.DG_current_count = self.DG_total_count.copy()

    def sample(self,k=1):
        total_size = np.sum(self.DG_total_count)
        result = []
        for try_index in range(k):
            thre = np.random.uniform(0,total_size)
            sum=0
            for i in range(len(self.DG_total_count)):
                sum += self.DG_total_count[i]
                if sum <= thre:
                    result.append(i)
                    break
        return np.array(result)

    def __repr__(self):
        return "Current G:" + np.array2string(self.DG_current_count)+"\n" + "Total G:" + np.array2string(self.DG_total_count)
