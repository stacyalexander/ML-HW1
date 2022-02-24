load EEG_driving_data_sample.mat;
sampleRate = 250;
% Total epochs
epochs = 1000; 
% Data Length is [{pwr_theta[1:8]}{pwr_alpha[9:19]}{label[20]}]
dataLength = 20;
labeled_data0 = zeros(epochs,dataLength);
labeled_data1 = zeros(epochs,dataLength);
for i = 1:1000
    data = extract_channel_features(data_class_1(:,:,i),250);
    labeled_data1(i,1:19) = data;
    labeled_data1(i,20) = 1;
end

for i = 1:1000
    data = extract_channel_features(data_class_0(:,:,i),250);
    labeled_data0(i,1:19) = data;
    labeled_data0(i,20) = 0;
end

% Write labeled data so CSV file to process in Python
csvwrite('label_0.csv',labeled_data0);
csvwrite('label_1.csv',labeled_data1);