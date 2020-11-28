import numpy as np
import matplotlib.pyplot as plt
import timeit
import sys

def oneHotLabel(x):
    make = np.zeros((10), dtype = float)
    make[x] = 1.0
    make = np.matrix(make)
    make = np.transpose(make)
    return make

def format(s):
    
    s = s.replace('[','')
    s = s.replace(']','')
    dataset = s.split()

    for i in range(len(dataset)):
        dataset[i] = float(dataset[i])
    
    
    data = np.array(dataset).reshape(-1,784)
    data = np.transpose(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    data = (data-mean)/std_dev
    return data

def sigmoid(x):
    return 1/(1+np.exp(-x))

def feedForward(s,w):
    w_input = w.dot(s)
    activations = sigmoid(w_input)
    return activations

def crossEntropy(target,output):
    #return np.sum(-target*np.log(output)-(1-target)*np.log(1-output))
    return(target-output)

def calcDeltas(w,a,e):
    delta = np.multiply(a, 1-a)
    delta2 = np.dot(w,e)
    delta_final = np.multiply(delta,delta2)
    return delta_final

def saveWeights(rand_weights, weights_):
    final_weights = open('netWeights.txt', 'w+')
    np.savetxt(final_weights, rand_weights)
    np.savetxt(final_weights, weights_)
    final_weights.close()

def savePlot(x,y):
    plt.plot(x,y)
    plt.xlabel("Execution Times")
    plt.ylabel("Learning accuracy")
    plt.title("Learning Rate:0.5")
    #plt.savefig('Graph_1')
    plt.show()





def main():
     
    my_input = sys.argv[1]
    if my_input == 'train':
        epoch = 1
        data = 0
        #ASSISGN WEIGHTS TO THE HIDDEN AND OUTPUT LAYER 
        rand_weights = np.random.uniform(low=-1, high=1, size = (30,784))
        weights_ = np.random.uniform(low=-1, high=1, size = (10,30))
        empty_string = ''
        input_empty_string = ''
        #FOR 2 EPOCHS 
        e_times = []
        l_accur = []
        for e in range(0,2):
            start = timeit.default_timer()
            count = 0.0
            accuracy = 0.0
            t_data = sys.argv[2]
            t_label = sys.argv[3]

            train_data = open(t_data, 'r')
            train_labels = open(t_label, 'r')
        
            while data<60000:
                # READ THE DATA OF TRAIN AND CONVERT TO ARRAY 
                for i in range(44):

                    empty_string = train_data.readline()
                    input_empty_string = input_empty_string+empty_string

                sam = format(input_empty_string)/float(255)
                # FEED FORWARDING 
                activations = feedForward(sam,rand_weights)
                output      = feedForward(activations,weights_)
                # CALCULATE ERRORS 
                label = train_labels.readline()
                label = int(label)
                target = oneHotLabel(label)
                error = crossEntropy(target,output)
                #BACK PROPAGATION  AND UPDATE WEIGHTS 
                weights_ = np.transpose(weights_)
                
                delta_3 = calcDeltas(weights_, activations, error)
                learning_rate = float(sys.argv[4])
                activations = np.transpose(activations)
                weights_ = np.transpose(weights_)
                weights_ +=  learning_rate*np.dot(error, activations)
                sam = np.transpose(sam)
                rand_weights +=  learning_rate*np.dot(delta_3, sam)
            
                data = data + 1
                input_empty_string = ''
                mode = np.argmax(output)
                if mode == label:
                    count = count + 1
            #RESET
            data = 0
            accuracy = (count/60000.0)*100.0
            l_accur.append(accuracy)
            #print(accuracy)
            print("#### TRAINING DATA #######")
            print('Epoch Number', epoch, 'Accuracy: ', accuracy, "%", "Error:", 100-accuracy, "%")
            #print('Epoch Number', epoch, "Error: ", 100-accuracy, "%")

            epoch = epoch + 1
            train_data.close()
            train_labels.close()
            stop = timeit.default_timer()
            time_taken = stop - start
            e_times.append(time_taken)
            #print(e_times)
            print("Execution time: ", time_taken, "s")

    

        savePlot(e_times,l_accur)
        saveWeights(rand_weights, weights_)

    elif my_input == "test":
        test = open(sys.argv[2], 'r')
        test_label = open(sys.argv[3], 'r')
        data = 0
        counter = 0.0
        empty_string = ''
        input_empty_string = ''

        w1= np.genfromtxt(sys.argv[4], skip_footer = 10)
        w2 = np.genfromtxt(sys.argv[4], skip_header = 30)

        while data<10000:
            for i in range(44):
                empty_string = test.readline()
                input_empty_string = input_empty_string+empty_string

            s = format(input_empty_string)/float(255)

            #weight_input = w1.dot(s)
            activations = feedForward(s,w1)
            output      = feedForward(activations,w2)
            data = data + 1
            lab = test_label.readline()
            lab = int(lab)
            input_empty_string = ''
            maximum = np.argmax(output)
            if maximum == lab:
                counter+=1.0
        accuracy = (counter/10000.0)*100.0
        print ('Test Accuracy:' , accuracy, "%")
         

if __name__ == "__main__":
    main()