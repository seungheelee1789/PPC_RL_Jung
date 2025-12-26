clear; clc; close all
load Fig2_representative_neurons_PSTH.mat

colortype2(1,:) = [0, 0, 255];       % R1-5kHz : blue / in case of IC, R1-Hit
colortype2(2,:) = [255, 0, 0];       % R1-10kHz: red / in case if IC, R2-CR
colortype2(3,:) = [246, 110, 199];   % R2-10kHz:light pink / in case of IC, R2-Hit
colortype2(4,:) = [91, 144, 246];    % R2-5kHz :light blue / in case of IC, R2-CR
cell_ID = {'AC #48','AC #194','IC #119'}; 
datalabelset = {'R1-5kHz', 'R1-10kHz', 'R2-10kHz', 'R2-5kHz'};
ccc = colortype2/255;
timescale = [-1*pre2:post2];

cri = 100; % gaussian smoothing 

for i = 1:size(eg_PSTH,1)
    figure()
    set(gcf,'Position',[100+20*i 300+20*i 200 100]);
    title(cell_ID(i))
    hold on
    ylabel(['Firing rate (Hz)'])
    for j = 1:size(eg_PSTH,2)
        aaa = smoothdata(mean(eg_PSTH{i,j},1)*1000,'gaussian', cri);
        plot(timescale, aaa, 'Color', ccc(j,:), 'Linewidth', 1)
    end
   xlim([-1000 4000])
end
