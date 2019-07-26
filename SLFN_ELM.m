function[train_acc,test_acc,train_time,test_time,beta,w]=SLFN_ELM(train_data,train_label,test_data,test_label,Num_hidden,reg_or_cl,p,q,varargin)
%SLFN_ELM - Single hidden layer feed forward network with Extreme Learning Machine 
%
% Syntax:  [train_acc,test_acc,beta,w]=SLFN_ELM(train_data,train_label,test_data,test_label,Num_hidden,p,q,varargin)
%
% Inputs:
%    train_data - Sample for training Network of size M1 x N
%    train_label - Class labels s for training size M1 x class_size (one hot vector)
%    test_data - Test samples of size M2 x N
%    test_label - Class labels for M2 x class_size (one hot vector)
%    Num_hidden - Number of nodes in hidden layer.
%    p - Random node intialization type and takes the following value
%         'rand(0,1)' : For uniformly distributed random intizlization ranges from 0 to 1.
%         'rand(-1,1)': For uniformly distributed random intizlization ranges from -1 to 1.
%         'ortho': Random Ortogonal vector intialization.
%         'xavier': Random weights in gaussian distribution with zero mean and (2/(N_in+N_out))
%         'relu' : Random weights in ran
%         Default is 'rand'
%    q - Activation function used for hidden layer. It takes folloing value
%         'None': No activation function.
%         'Relu': Rectilinear activation function (Ref: https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
%         'Sigmoid' or 'Logistic' : Sigmoid activation function (Ref: http://mathworld.wolfram.com/SigmoidFunction.html)
%         'Tanh' : Hyparabolic tan function.
%         'Softsign' : converges polynomially instead of exponentially towards its asymptotes.
%         'Softplus': A smooth version of the ReLu
%         'Sin' : Sinusoidal Function
%         'Cos' : Co-Sin function
%         'Sinc' : cardinal sine function
%         'LeakyRelu' : Leaky rectified linear unit  wir alpah 0.001
%         'Gaussian' : Gaussian distribution.
%         'BentIde' : Bent Identity Function
%         'ArcTan' : Tan inverse function.
%         Default is 'Sigmoid'
% Outputs:
%     train_acc- Accuracy of the network on training set
%     test_acc- Accuracy of the network on testing set
%     beta- Moore-penrose inverse approximation of hidden to output weight
%     w- Randomly assigned Weight to input to hidden nodes
%
% Example: 
%     [train_acc,test_acc,beta,w]=SLFN_ELM(Train_image,Train_label,Test_image,Test_label,30);
%     [train_acc,test_acc,beta,w]=SLFN_ELM(Train_image,Train_label,Test_image,Test_label,30,'ortho');
%     [train_acc,test_acc,beta,w]=SLFN_ELM(Train_image,Train_label,Test_image,Test_label,30,'ortho','Sigmoid');
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: *****************
%
%
% Reference: (Bibtex)
% @article{huang2006extreme,
%   title={Extreme learning machine: theory and applications},
%   author={Huang, Guang-Bin and Zhu, Qin-Yu and Siew, Chee-Kheong},
%   journal={Neurocomputing},
%   volume={70},
%   number={1-3},
%   pages={489--501},
%   year={2006},
%   publisher={Elsevier}
% }
% 
% 
% Author: Dibyasundar Das, Ph.D., Computer Science,
% National Institute of Technology Rourkela, Odisha, India.
% email address: dibyasundar@ieee.org
% July 2019; Last revision: 06-July-2019

%------------- BEGIN CODE --------------
train_acc=NaN;test_acc=NaN;beta=NaN;w=NaN;
[~,sz_sample]=size(train_data);
bias_sz=sz_sample+1;
if nargin<=6
    p='rand(0,1)';
end
if nargin<=7
    q='Relu';
end

switch p
    case 'rand(0,1)'
        w=rand(bias_sz,Num_hidden);
    case 'rand(-1,1)'
        w=-1+(rand(bias_sz,Num_hidden)*2);
    case 'xavier'
        w=normrnd(0,(2/(size(train_data,2)+size(test_data,2))),bias_sz,Num_hidden);
    case 'relu'
        w=normrnd(0,(2/(Num_hidden)),bias_sz,Num_hidden);
    case 'ortho'
        w=rand(bias_sz,Num_hidden);
        [s,~,d]=svd(w);
        if bias_sz>=Num_hidden
            w=s(1:bias_sz,1:Num_hidden);
        else
            w=d(1:bias_sz,1:Num_hidden);
        end
    otherwise
        disp(strcat('Unkown input .....',{' '},p,' is not a valid input.... recheck input'));return;
end
%Training in ELM
tic
I=train_data;
[~,n]=size(I);
I(:,n+1)=1;
H=(I*w);
H=activation_fun(H,q);
beta=pinv(H)*train_label;
train_time=toc;
%Training accuracy
%Testing in ELM

I=train_data;
[m,n]=size(I);
I(:,n+1)=1;
H=(I*w);
H=activation_fun(H,q);
obt_label=H*beta;

if reg_or_cl==0
    [~,pos1]=max(obt_label,[],2);
    [~,pos2]=max(train_label,[],2);
    train_acc=size(find(pos1==pos2),1)./size(train_label,1);
else
    train_acc=sqrt(mean((obt_label-train_label).^2));
end



%Testing in ELM
tic
I=test_data;
[m,n]=size(I);
I(:,n+1)=1;
H=(I*w);
H=activation_fun(H,q);
obt_label=H*beta;
test_time=toc;
if reg_or_cl==0
    [~,pos1]=max(obt_label,[],2);
    [~,pos2]=max(test_label,[],2);
    test_acc=size(find(pos1==pos2),1)./size(test_label,1);
else
    test_acc=sqrt(mean((obt_label-test_label).^2));
end
end
%% Function for Activation function
function[x]=activation_fun(x,type)
    switch type
        case 'Relu'
            x(x<0)=0;
        case 'Sigmoid'
            x=1./(1+exp(-x));
        case 'Tanh'
            x=tanh(x);
        case 'Softsign'
            x=(x)./(1+abs(x));
        case 'Softplus'
            x=log(1+exp(x));
        case 'Sin'
            x=sin(x);
        case 'Cos'
            x=cos(x);
        case 'Sinc'
            x(x==0)=1;
            x(x~=0)=sin(x(x~=0))./x(x~=0);
        case 'LeakyRelu'
            x(x<0)=0.001.*(x(x<0));
        case 'Logistic'
            x=1./(1+exp(-x));
        case 'Gaussian'
            x=exp(-(x.^2));
        case 'BentIde'
            x=((sqrt((x.^2)+1)-1)./2)+x;
        case 'ArcTan'
            x=atan(x);
        case 'None'
            x=x;
        otherwise
            disp(strcat('Unkown input .....',{' '},type,' is not a valid input.... recheck input'));return;
    end
end
%------------- END OF CODE --------------