filename = 'EEG_driving_data_sample.mat';
myVars = {'data_class_0','data_class_1', 'number_epochs'};
S = load(filename,myVars{:});
sampling_rate = 250;
for i = 1:1000
    class0(:,i) = extract_channel_features(S.data_class_0(:,:,i), sampling_rate);
end

for i = 1:1000
    class1(:,i) = extract_channel_features(S.data_class_1(:,:,i), sampling_rate);
end

class0 = [class0',zeros(1000,1)];
class1 = [class1',ones(1000,1)];
csvwrite('class_label_0.csv',class0)
csvwrite('class_label_1.csv',class1)