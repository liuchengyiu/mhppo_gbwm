import torch
import numpy as np
import matplotlib.pyplot as plt
 
TIME_STEP=10
INPUT_SIZE=1
HIDDEN_SIZE=32
LR=0.02
 
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        
        self.rnn=torch.nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True)
#设置batch_first为True,那么输入数据的维度为(batch_size,time_step,input_size)
#如果不设置这个值，或者设置为False，那么输入数据的维度为(time_step,batch_size,input_size)
 
        self.out=torch.nn.Linear(HIDDEN_SIZE,1)
    #将隐藏层输出转化为需要的输出
        
    
    def forward(self,x,h_state):
        #因为在RNN中，下一个时间片隐藏层状态的计算需要上一个时间片的隐藏层状态，所以我们要一直传递这个h_state
        #x (batch_size,time_step,INPUT_SIZE)
        
        r_out,h_state=self.rnn(x,h_state)
        #h_state也要作为RNN的一个输入和一个输出
        #r_out：(batch_size,time_step,HIDDEN_SIZE)
        #h_state：(batch_size,time_step,HIDDEN_SIZE)
        
        outs=[]
        
        for time_step in range(r_out.size()[1]):
            outs.append(self.out(r_out[:,time_step,:]))
            #每一个要被self.out运算的元素[batch_size,1,HIDDEN_SIZE]
            #每个计算完，被append到outs的元素[batch_size,1,1]
        return torch.stack(outs,dim=1),h_state
    #返回的第一个元素[batch_size,time_step,1]
    #torch.stack函数的维度和axis不一样，dim=1的意思是在第一个维度处叠加
    
rnn=RNN()
print(rnn)
'''
RNN(
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear(in_features=32, out_features=1, bias=True)
)
'''
 
 
optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=torch.nn.MSELoss()
 
 
h_state=None
 
for step in range(100):
    start=step*np.pi
    end=(step+1)*np.pi
    
    steps=np.linspace(start,end,TIME_STEP,dtype=np.float32)
    #这里dtype这一部分一定要加，不然的话会报错，RuntimeError: expected scalar type Double but found Float
    
    x_np=np.sin(steps).reshape(1,TIME_STEP,INPUT_SIZE)
    y_np=np.cos(steps).reshape(1,TIME_STEP,1)
    #目标：用sin预测cos
    
    x=torch.from_numpy(x_np)
    y=torch.from_numpy(y_np)
    
    prediction,h_state=rnn(x,h_state)
    #每一组input，都对应了一个h_state和一个prediction
    
    h_state=h_state.data
    #将对应的h_state向后传
    
    loss=loss_func(prediction,y)
    
    optimizer.zero_grad()
    #清空上一步的参与更新参数值
    
    loss.backward()
    #误差反向传播，计算参数更新值
    
    optimizer.step()
    #将参数更新值施加到rnn的parameters上
    
    if(step % 10==0):
        plt.plot(steps,prediction.data.numpy().flatten(),'g*')
        plt.plot(steps,y_np.flatten(),'r-')
        plt.show()