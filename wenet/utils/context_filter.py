import math
import copy
import sys
import torch
import time

class PosteriorFilter:
    def __init__(self,):
        pass
    
    def posterior_filter(self, posterior, context_list, context_lengths, context_score):
        if len(posterior) == 0:
            return 
        max_p, _ = torch.max(posterior, dim=0, keepdim=False)
        max_p = max_p.tolist()
        for i in range(1, len(context_list)):
            score = 0
            for j in range(context_lengths[i]):
                score += max_p[context_list[i][j]]
            score /= context_lengths[i]
            context_score[i] = max(context_score[i], score)


class ContextFilter:
    def __init__(self,
        context_list,
        context_lengths,
        window_size = 64,
        topk_first = 50,
        topk_second = -3,
    ):
        self.context_list = context_list
        self.context_lengths = context_lengths
        
        self.context_score = {}
        for i in range(1, len(context_list)):
            self.context_score[i] = -float('inf')
        
        self.posterior_filter_ = PosteriorFilter()
        self.window_size = window_size
        self.topk_first = topk_first
        self.topk_second = topk_second


    def posterior_filter(self, posterior):
        self.posterior_filter_.posterior_filter(posterior, self.context_list, self.context_lengths, self.context_score)

    def second_filter(self, posterior):
        topk_list = sorted(self.context_score.items(), key=lambda x:x[1], reverse=True)
        topk = min(self.topk_first, len(topk_list))

        topk_score = {}

        topk_list = topk_list[:topk]
        for i in range(topk):
            topk_list[i] = topk_list[i][0]
            topk_score[topk_list[i]] = -float('inf')

        start = 0
        end = min(self.window_size, len(posterior))
        while True:
            n = end - start
            posterior_win = posterior[start: end]
            for i in topk_list:
                if i == 0:
                    continue
                m = self.context_lengths[i]
                if m > n:
                    continue
                dp = [[0] * m for _ in range(n)]
                
                for k in range(m):
                    for j in range(k, n):
                        if k == 0:
                            if j == 0:
                                dp[j][k] = posterior_win[j][self.context_list[i][k]]
                            else :
                                dp[j][k] = max(dp[j - 1][k], posterior_win[j][self.context_list[i][k]])
                        else:
                            if j == k:
                                dp[j][k] = dp[j - 1][k - 1] + posterior_win[j][self.context_list[i][k]]
                            else :
                                dp[j][k] = max(dp[j - 1][k - 1] + posterior_win[j][self.context_list[i][k]], dp[j - 1][k])
                topk_score[i] = max(topk_score[i], dp[-1][-1] / m)
            
            if end == len(posterior):
                break
            start += self.window_size // 4
            end += self.window_size // 4
            if end > len(posterior):
                end = len(posterior)
                start = end - self.window_size

        topk_list = sorted(topk_score.items(), key=lambda x:x[1], reverse=True)
        # topk = min(self.topk_second, len(topk_list))
        res_list = [self.context_list[0]]
        res_lengths = [1]

        for i in range(len(topk_list)):
            if topk_list[i][1] < self.topk_second:
                break
            res_list.append(self.context_list[topk_list[i][0]])
            res_lengths.append(self.context_lengths[topk_list[i][0]])
        return res_list, res_lengths