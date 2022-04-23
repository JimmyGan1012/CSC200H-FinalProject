import operator
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
To do list:
1. modify all add to target dataset
Have the add function return status of (partially fail, fail, success) instead of true/false (Done)
2. modify reward function update to take in accepted count
3. update argmax k i function
"""


class Datasets:

    def __init__(self):
        self.DS = []
        self.N = 0
        self.Gs = None

    def create_type1_data(self):
        DG_distribution = np.array([0.1, 0.3, 0.6])
        tuple_len = 10000
        self.Gs = np.arange(0, 3)
        for i in range(10):
            # construct a dataset
            dataset_idx = i
            tuples = []
            for tuple_idx in range(tuple_len):
                tup_demo = random.choices(self.Gs, DG_distribution, k=1)[0]
                tup = (np.random.uniform(0, 1), tup_demo)
                tuples.append(tup)

            # C0 = np.random.randint(low=1,high=10,size=1)[0] # unequal cost
            C0 = 1
            Cx = np.random.uniform()
            self.DS.append(MaryDataset(dataset_idx, tuples, self.Gs, C0,Cx))
        self.N = 10

    # unequal cost
    def create_type2_data(self):
        DG_distribution = np.array([0.1, 0.3, 0.6])
        tuple_len = 1000
        self.Gs = np.arange(0, 3)
        for i in range(10):
            # construct a dataset
            dataset_idx = i
            tuples = []
            for tuple_idx in range(tuple_len):
                tup_demo = random.choices(self.Gs, DG_distribution, k=1)[0]
                tup = (np.random.uniform(0, 1), tup_demo)
                tuples.append(tup)

            # C0 = np.random.randint(low=1,high=10,size=1)[0] # unequal cost
            C0 = 1
            Cx = np.random.uniform()
            self.DS.append(MaryDataset(dataset_idx, tuples, self.Gs, C0,Cx))
        self.N = 10


class MaryDataset:

    def __init__(self, i, tuples, Gs, C0, Cx):
        self.id = i
        self.tuples = tuples
        self.N = len(tuples)
        # number of samples taken from each group
        self.Ts = {j: 0 for j in Gs}
        self.Gs = Gs
        self.C = lambda x: C0 + Cx * (x-1)
        # number of samples taken so far
        self.t = 0
        self.seen = dict()
        # true probs for computing regret
        self.Ns = {j: 0 for j in Gs}
        for i, v in enumerate(tuples):
            self.Ns[v[1]] += 1
        self.Ps = {j: float(self.Ns[j]) / self.N for j in Gs}
        # true cost value for computing purpose
        self.C0 = C0
        self.Cx = Cx

    def sample(self, k=1):
        """
        sample k samples from dataset :param k: :return: None if all samples are seen, otherwise return the sample s,
        consisting list of tuples of length, 1 <= length <= k
        """
        if k == 1:
            idx = random.randint(0, self.N - 1)
            s = Sample([self.tuples[idx]], self.id, self.C(k))
        else:
            sample_tuples = []
            for j in range(k):
                idx = random.randint(0, self.N - 1)
                sample_tuples.append(self.tuples[idx])
            s = Sample(sample_tuples, self.id, self.C(k))

        # two cases:
        # 1. samples partially/completely unseen
        # 2. all have seen
        removal_idx = []
        for record_idx in range(len(s.rec)):
            if s.rec[record_idx][0] in self.seen:
                removal_idx.append(record_idx)
        # if all sampled seen, return None
        if len(removal_idx) == len(s.rec):
            return None
        # otherwise, remove seen elements
        for idx in reversed(removal_idx):
            s.rec.pop(idx)
        # update seen elements
        for idx in range(len(s.rec)):
            self.seen[s.rec[idx][0]] = True
        return s

    def update_stats(self, s):
        for record in s.rec:
            j = record[1]
            self.Ts[j] += 1


class MaryTarget:
    tuples = []

    def __init__(self, Gs, Qs):
        self.Qs = Qs
        self.Gs = Gs
        self.Os = {j: 0 for j in Gs}

    def add(self, sample):
        """
        add the sample to the target dataset, check if sample met demo requirement
        :param sample: Sample object that has list of tuples, dataset id and cost
        :return: return the number of samples that satisfy requirement
        """
        accepted_count = 0
        for record in sample.rec:
            demo_group = record[1]
            if self.Os[demo_group] < self.Qs[demo_group]:
                # accept
                self.tuples.append(record)
                self.Os[demo_group] += 1
                accepted_count += 1
        return accepted_count

    def complete(self):
        for j in self.Gs:
            if self.Os[j] != self.Qs[j]:
                return False
        return True


class Sample:
    def __init__(self, rec, dataset_id, cost):
        """
        create a sample object
        :param rec: array-like records with shape(#of tuples, 1), each entry is a tuple [sample, demographic group]
        :param dataset_id: identifier of sampled dataset
        :param cost: cost
        """
        self.rec = rec
        self.dataset_id = dataset_id
        self.cost = cost


class UnknownDT:

    def __init__(self, ds, target, Gs, ps=None, budget=10000):
        self.datasets = ds
        self.target = target
        self.Gs = Gs
        # number of samples taken so far
        self.t = 0
        # underlying distribution
        if ps is None:
            self.get_underlying_dist()
        else:
            self.Ps = ps
        self.budget = budget
        self.maximum_k = 20

    def get_underlying_dist(self):
        counts = {j: 0 for j in self.Gs}
        for i in range(len(self.datasets.DS)):
            for s in self.datasets.DS[i].tuples:
                counts[s[1]] += 1
        csum = sum([len(d.tuples) for d in self.datasets.DS])
        self.Ps = {j: float(c) / csum for j, c in counts.items()}

    def select_dataset(self):
        rewards = dict()
        for i in range(len(self.datasets.DS)):
            # reward of a dataset
            rewards[i] = self.get_reward(i)
            # upper bound based on UCB strategy and Hoeffding’s Inequality
            ub = self.get_upper_bound(i,1)
            rewards[i] += ub
        Dl = max(rewards.items(), key=operator.itemgetter(1))[0]
        return Dl

    def k_accepted_prob(self,i,k):
        N_i = self.datasets.DS[i].N
        O_i = self.datasets.DS[i].t
        unseen_prob = np.power(((N_i-O_i)/N_i),k)
        product = 0.0
        for j in self.Gs:
            product += 1-((self.target.Os[j]/self.target.Qs[j]) * self.Ps[j])
        not_overflow_prob = np.power(product, k)
        return unseen_prob*not_overflow_prob

    def estimate_reward(self,i,k=1):
        expected_accept = 0.0
        for g in range(1,k):
            expected_accept += g * self.k_accepted_prob(i,g)
        return sum(
            [float(self.datasets.DS[i].Ts[j]) / (self.Ps[j] * self.datasets.DS[i].C(k) * self.datasets.DS[i].t) for j in
             self.Gs if
             self.target.Os[j] < self.target.Qs[j]]) * expected_accept

    def real_reward(self,i,k,accepted_count):
        return sum(
            [float(self.datasets.DS[i].Ts[j]) / (self.Ps[j] * self.datasets.DS[i].C(k) * self.datasets.DS[i].t) for j in
             self.Gs if
             self.target.Os[j] < self.target.Qs[j]]) * accepted_count

    def get_reward(self, i,k=1):
        """
        :param i: dataset index
        :param k: number of queries
        :return: reward estimation
        """
        return sum(
            [float(self.datasets.DS[i].Ts[j]) / (self.Ps[j] * self.datasets.DS[i].C(k) * self.datasets.DS[i].t) for j in
             self.Gs if
             self.target.Os[j] < self.target.Qs[j]])

    def select_dataset_k(self):
        """
        :return: dataset_index i, number of query tuples k
        """
        rewards = dict()
        for i in range(len(self.datasets.DS)):
            for k in range(self.maximum_k):
                if k==0: continue # force sample at least one sample
                # reward of a dataset
                rewards[(i,k)] = self.estimate_reward(i,k)
                # upper bound based on UCB strategy and Hoeffding’s Inequality
                ub = self.get_upper_bound(i,k)
                rewards[(i,k)] += ub
        Dl,k = max(rewards.items(), key=operator.itemgetter(1))[0]
        return Dl,k

    def get_upper_bound(self, i,k):
        lower_bound = 0.0
        upper_bound = max([self.Ps[j] / self.datasets.DS[i].C(k) for j in self.Gs if self.target.Qs[j] > self.target.Os[j]])
        return (upper_bound-lower_bound) * math.sqrt(2.0 * math.log(self.t) / self.datasets.DS[i].t)

    def first_round_sample(self):
        rewards = []
        cost = 0
        for Dl in range(len(self.datasets.DS)):
            # sample one sample from every dataset
            Ol = self.datasets.DS[Dl].sample(k=1)
            # update the total number of samples
            self.t += 1
            self.datasets.DS[Dl].t += 1
            if Ol is not None:
                # update only when part or all sample has not been seen
                # update the taken sample count in dataset
                self.datasets.DS[Dl].update_stats(Ol)
                # add the sample to target dataset
                accepted_count = self.target.add(Ol)
                if accepted_count > 0:
                    rewards.append(self.real_reward(Dl,k=1,accepted_count=accepted_count))
                else:
                    rewards.append(0.0)
            # update cost to add the cost to sample exactly one record
            cost += self.datasets.DS[Dl].C(1)
        return rewards, cost

    def run_ucb_baseline(self):
        history_choice = []
        history_k = []

        progress = []
        terminate = False
        dupsamples = 0
        overflow = 0
        rewards, cost = self.first_round_sample()
        print("cost of first round sampling",cost)
        # consider one round of sampling datasets
        num_sampled = len(self.datasets.DS)
        if self.target.complete():
            print("terminated first round")
            terminate = True
        while num_sampled < self.budget and not terminate:
            Dl = self.select_dataset()
            k=1
            # print("already sampled {} tuples".format(self.t))
            # print('selecting {} dataset'.format(Dl))
            Ol = self.datasets.DS[Dl].sample(k=k)
            # update the total number of samples
            self.t += k
            self.datasets.DS[Dl].t += k
            if Ol is None:
                # if all samples are seen
                dupsamples += k
            else:
                # update only when the samples are partially seen or completely unseen
                # len(Ol.rec) is number of unseen sample returned
                num_tuples_sampled = len(Ol.rec)
                dupsamples += k - num_tuples_sampled
                # update dataset statistics and add samples to target dataset
                self.datasets.DS[Dl].update_stats(Ol)
                accepted_count = self.target.add(Ol)
                if accepted_count > 0:
                    # demographic groups count not satisfied
                    rewards.append(self.real_reward(Dl,k, accepted_count))
                    overflow += num_tuples_sampled - accepted_count
                else:
                    rewards.append(0.0)
                    overflow += num_tuples_sampled

            history_choice.append(Dl)
            history_k.append(k)

            cost += self.datasets.DS[Dl].C(k)
            num_sampled += k
            if self.target.complete():
                terminate = True
        if not terminate:
            print('timeout')
        if terminate:
            print('cost: %f, num_sampled: %d, dupsamples: %d, overflowed samples: %d' % (cost, num_sampled, dupsamples, overflow))
        return np.array(history_choice), np.array(history_k)

    def run_ucb(self):
        history_choice = []
        history_k = []

        progress = []
        terminate = False
        dupsamples = 0
        overflow = 0
        rewards, cost = self.first_round_sample()
        print("cost of first round sampling",cost)
        # consider one round of sampling datasets
        num_sampled = len(self.datasets.DS)
        if self.target.complete():
            print("terminated first round")
            terminate = True
        while num_sampled < self.budget and not terminate:
            Dl, k = self.select_dataset_k()
            # print("already sampled {} tuples".format(self.t))
            # print('selecting {} dataset and {} number of query'.format(Dl,k))
            Ol = self.datasets.DS[Dl].sample(k=k)
            # update the total number of samples
            self.t += k
            self.datasets.DS[Dl].t += k
            if Ol is None:
                # if all samples are seen
                dupsamples += k
            else:
                # update only when the samples are partially seen or completely unseen
                # len(Ol.rec) is number of unseen sample returned
                num_tuples_sampled = len(Ol.rec)
                dupsamples += k - num_tuples_sampled
                # update dataset statistics and add samples to target dataset
                self.datasets.DS[Dl].update_stats(Ol)
                accepted_count = self.target.add(Ol)
                if accepted_count > 0:
                    # demographic groups count not satisfied
                    rewards.append(self.real_reward(Dl,k, accepted_count))
                    overflow += num_tuples_sampled - accepted_count
                else:
                    rewards.append(0.0)
                    overflow += num_tuples_sampled

            history_choice.append(Dl)
            history_k.append(k)

            cost += self.datasets.DS[Dl].C(k)
            num_sampled += k
            if self.target.complete():
                terminate = True
        if not terminate:
            print('timeout')
        if terminate:
            print('cost: %f, num_sampled: %d, dupsamples: %d, overflowed samples: %d' % (cost, num_sampled, dupsamples, overflow))
        return np.array(history_choice), np.array(history_k)

    def select_dataset_cost(self):
        scores = {i: 1.0 / self.datasets.DS[i].C for i in range(len(self.datasets.DS))}
        ssum = sum(list(scores.values()))
        ps, ls = [], []
        for l, s in scores.items():
            ls.append(l)
            ps.append(s / ssum)
        return random.choices(ls, ps, k=1)[0]


if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)

    # count = 0
    # for maryData in datasets.DS:
    #     print("Data set ",count)
    #     count+=1
    #     print("Cost = {} + {} * k".format(maryData.C0, maryData.Cx))
    #     # print(maryData.tuples)



    # # baseline performance
    datasets = Datasets()
    datasets.create_type1_data()
    target = MaryTarget(datasets.Gs, [10, 10, 10])
    unknwonDT_base = UnknownDT(datasets, target, datasets.Gs)
    history_choice_base,history_k_base = unknwonDT_base.run_ucb_baseline()

    np.random.seed(1)
    random.seed(1)

    # k_performace
    datasets_k = Datasets()
    datasets_k.create_type1_data()
    target_k = MaryTarget(datasets_k.Gs, [10, 10, 10])
    unknwonDT_k = UnknownDT(datasets_k, target_k, datasets_k.Gs)

    history_choice, history_k = unknwonDT_k.run_ucb()

    # plt.figure()
    # plt.plot(history_choice_base, 'x')
    # #
    # plt.figure()
    # plt.plot(history_choice, 'x')
    # plt.show()
