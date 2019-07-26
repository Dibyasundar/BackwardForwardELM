function[]= mnist_prog()
    clear;clc;close all;
    if ~exist('train-images-idx3-ubyte.gz','file')
        url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
        file_name='train-images-idx3-ubyte.gz';
        websave(file_name,url);
    end
    if ~exist('train-labels-idx1-ubyte.gz','file')
        url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
        file_name='train-labels-idx1-ubyte.gz';
        websave(file_name,url);
    end
    if ~exist('t10k-images-idx3-ubyte.gz','file')
        url='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
        file_name='t10k-images-idx3-ubyte.gz';
        websave(file_name,url);
    end
    if ~exist('t10k-labels-idx1-ubyte.gz','file')
        url='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz';
        file_name='t10k-labels-idx1-ubyte.gz';
        websave(file_name,url);
    end
    gunzip('train-images-idx3-ubyte.gz')
    gunzip('train-labels-idx1-ubyte.gz')
    gunzip('t10k-images-idx3-ubyte.gz')
    gunzip('t10k-labels-idx1-ubyte.gz')
    
    
    train_images=loadMNISTImages('train-images-idx3-ubyte');
    train_labels=loadMNISTLabels('train-labels-idx1-ubyte');
    test_images=loadMNISTImages('t10k-images-idx3-ubyte');
    test_labels=loadMNISTLabels('t10k-labels-idx1-ubyte');
    
    problem_type=0;% 0 for classification 1 for regression
    
    %%% number of hidden nodes
    h_node=[1:20];
    %%% Weight initialization scheme
    h_rand={'relu'};%{'ortho','rand(0,1)','rand(-1,1)','xavier','relu'}; 
    %%% Activation funtion
    h_acti={'Sigmoid'};%{'None','Relu','Sigmoid','Tanh','Softsign','Sin','Cos','LeakyRelu','BentIde','Gaussian','ArcTan'}; 
    fprintf('%s\n','____________________________________________________________________________________________________________________________________________________________________________________');
    fprintf('|%15s|%20s|%15s|%20s|%20s|%20s|%20s|%20s|%20s|\n','Weight init','Activation fun','Num node', 'ELM train time', 'ELM test time', 'ELM test Acc.', 'BF-ELM train time', 'BF-ELM test time', 'BF-ELM test Acc.');
    fprintf('%s\n','____________________________________________________________________________________________________________________________________________________________________________________');
    for i=1:size(h_acti,2)
        for j=1:size(h_rand,2)
            for k=1:size(h_node,2)
                [train_acc1,test_acc1,train_time1,test_time1,beta1,w1]=SLFN_ELM(train_images,full(ind2vec(train_labels'+1))',test_images,full(ind2vec(test_labels'+1))',h_node(k),problem_type,h_rand{j},h_acti{i});
                [train_acc2,test_acc2,train_time2,test_time2,beta2,w2]=ELM_forward_backward(train_images,train_labels+1,test_images,test_labels+1,h_node(k),problem_type,h_rand{j},h_acti{i});
                fprintf('|%15s|%20s|%15d|%20.4f|%20.4f|%20.2f|%20.4f|%20.4f|%20.2f|\n',h_rand{j},h_acti{i},h_node(k),train_time1,test_time1,test_acc1*100, train_time2,test_time2,test_acc2*100);
            end
        end
    end
    fprintf('%s\n','_____________________________________________________________________________________________________________________________________________________________________________________');
end



function images = loadMNISTImages(filename)
    %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    %the raw MNIST images

    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);

    numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    images = permute(images,[2 1 3]);

    fclose(fp);

    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    % Convert to double and rescale to [0,1]
    images = double(images') / 255;

end
function labels = loadMNISTLabels(filename)
    %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
    %the labels for the MNIST images

    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);

    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);

    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

    labels = fread(fp, inf, 'unsigned char');

    assert(size(labels,1) == numLabels, 'Mismatch in label count');

    fclose(fp);

end
