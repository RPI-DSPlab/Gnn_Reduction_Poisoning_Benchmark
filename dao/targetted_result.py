import pandas as pd


class TargettedResultDAO():
    def __init__(self, df=None):
        if df is not None:
            self.result = {
                'seed': df['seed'].tolist(),
                'dataset': df['dataset'].tolist(),
                'method': df['method'].tolist(),
                'rate': df['rate'].tolist(),
                'misclassification_rate': df['misclassification_rate'].tolist(),
                'hard_asr': df['hard_asr'].tolist(),
                'middle_asr': df['middle_asr'].tolist(),
                'easy_asr': df['easy_asr'].tolist(),
                'clean_acc_avg': df['clean_acc_avg'].tolist(),
                'node_ratio_avg': df['node_ratio_avg'].tolist(),
                'edge_ratio_avg': df['edge_ratio_avg'].tolist(),
                'node_ratio_std': df['node_ratio_std'].tolist(),
                'edge_ratio_std': df['edge_ratio_std'].tolist(),
                'clean_acc_std': df['clean_acc_std'].tolist(),
            }
        else:
            self.result = {
                'seed': [],
                'dataset': [],
                'method': [],
                'rate': [],
                'misclassification_rate': [],
                'hard_asr': [],
                'middle_asr': [],
                'easy_asr': [],
                'clean_acc_avg': [],
                'node_ratio_avg': [],
                'edge_ratio_avg': [],
                'node_ratio_std': [],
                'edge_ratio_std': [],
                'clean_acc_std': [],
            }
    
    def __str__(self):
        return pd.DataFrame(self.result).__str__()
    
    def append(self, seed, dataset, method, rate, misclassification_rate, hard_asr, middle_asr, easy_asr, clean_acc_avg, node_ratio_avg, edge_ratio_avg, node_ratio_std, edge_ratio_std, clean_acc_std):
        self.result['seed'].append(seed)
        self.result['dataset'].append(dataset)
        self.result['method'].append(method)
        self.result['rate'].append(rate)
        self.result['misclassification_rate'].append(misclassification_rate)
        self.result['hard_asr'].append(hard_asr)
        self.result['middle_asr'].append(middle_asr)
        self.result['easy_asr'].append(easy_asr)
        self.result['clean_acc_avg'].append(clean_acc_avg)
        self.result['node_ratio_avg'].append(node_ratio_avg)
        self.result['edge_ratio_avg'].append(edge_ratio_avg)
        self.result['node_ratio_std'].append(node_ratio_std)
        self.result['edge_ratio_std'].append(edge_ratio_std)
        self.result['clean_acc_std'].append(clean_acc_std)
    
    def concat(self, other):
        for key in self.result.keys():
            self.result[key] += other.result[key]
    
    def export(self, path):
        df = pd.DataFrame(self.result)
        df.to_csv(path, index=False)