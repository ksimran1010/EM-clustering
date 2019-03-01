import numpy as np

class EMmethod :
    def __init__(self,k ,epoch , data):
        init_prior =  np.ones(shape = (k,1))*(1/k)         #p(b)
        gaussian_matrix =  np.random.choice([1,2,3],size = [2,k])
        self.k = k
        self.gaussian_matrix = gaussian_matrix
        self.init_prior = init_prior
        self.data = data
        self.epoch = epoch

    def cal_likelihood(self,data,gaus_mat):
        term1  = 1/ np.sqrt(2* np.pi * gaus_mat[1,:])
        term2 = np.exp((-(data- gaus_mat[0,:].reshape(self.k,1))^2)/ (2*np.power(gaus_mat[1,:].reshape(self.k,1),2)))
        return(np.multiply(term1.reshape(self.k,1),term2))

    def cal_posterior(self,prior,likehd):
        num= np.multiply(prior,likehd)      # p(xi|b)*p(b)
        denom = np.sum(num,axis= 0)
        return(np.divide(num,denom))

    def cal_mean(self,data, post):
        return(np.sum(np.multiply(data,post),axis= 1))

    def cal_variance(self, data, post,mean):
        numerator= np.sum(np.multiply(np.power(np.add(mean.reshape(self.k,1),post),2),post),axis = 1)
        denom  = np.sum(post)
        return(numerator/denom)

    def update_priors(self,post,len_data):
        return(np.divide (np.sum(post , axis = 1),len_data).reshape(post.shape[0],1))

    def cal_em(self):
        for iter in range(0 ,self.epoch):
            posterior = self.cal_posterior(self.init_prior,self.cal_likelihood(self.data,self.gaussian_matrix))
            self.gaussian_matrix[0,:] = self.cal_mean(self.data,posterior)
            self.gaussian_matrix[1,:]=self.cal_variance(self.data,posterior,self.gaussian_matrix[0,:])
            self.init_prior = self.update_priors(posterior,len(self.data))
        return self.gaussian_matrix


if __name__ == '__main__':
    data = np.random.randint(0,10 , size = 100)
    emObj= EMmethod(20,10,data)
    print( emObj.cal_em())

