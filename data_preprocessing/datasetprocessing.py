##some function for data extraction and downloading


#examples for collecting data 
## MOVE TO datapreprocessing dir
'''
samples = []
with open('./data/data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)
'''
'''
samples = []

#load csv file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)'''





#MOVE TO preprocessing dir
train_samples, validation_samples = train_test_split(samples,test_size=0.15) 
#simply splitting the dataset to train and validation set usking sklearn. .15 indicates 15% of the dataset is validation set
# the 15% needs to be defined according to research, just random atm


#code for generator
#takes raw data and performs simple transformation (flip, angle augmentation)
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while True: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size] 
            #handle that more data than rgb is collected

            images = []
            angles = []
            #check if we are collecting single camera image 
            for batch_sample in batch_samples:
                    name = 'format of file name tbd'
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
                    center_angle = float(batch_sample[0]) #getting the steering angle measurement
                    images.append(center_image)
                    angles.append(center_angle)
                   
                   
                        
                        # Code for Augmentation of data.
                        # We take the image and just flip it and negate the measurement
                        
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1)
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) #here we do not hold the values of X_train and y_train instead we yield the values which means we hold until the generator is running
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)