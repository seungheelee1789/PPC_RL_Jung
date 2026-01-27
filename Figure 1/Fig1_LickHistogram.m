%%%%%%%%%%%%%% Jung & Lee et al, Parietal top-down projections balance flexibility and stability in adaptive learning 
%%%%%%%%%%%%%% Eunji Jung, 2025-May-02
clear; clc; close all; 
load ExtendedDataFig1_LickHistogram.mat 

%%% Data strucrue %%%
% lick_stacked{Session, Trialtype}{trial, 1:5} 
%    : stim onset, lick during stmonset to r.w., 1st lick, outcome onset, lick -1 to 6 sec
% Trialtype: R1-Hit, R1-Miss, R1-FA, R1-CR, R2-Hit, R2-Miss, R2-FA, R2-CR

nSession = size(lick_stacked,1);
nType = size(lick_stacked,2);
lick.hist = cell(nSession, nType);
lick.first = cell(nSession, nType);
lick.outcome = cell(nSession, nType/2);
lick.stim = cell(nSession, nType/2);

for s = 1:nSession
    for j = 1:nType
        tttt = 0; tt = 0; ttt = 0;
        nTrial = size(lick_stacked{s,j},1);
        for t = 1:nTrial
            if ~isempty(lick_stacked{s,j}{t,2})
                tttt = tttt + 1;
                lick.hist{s,j}{tttt,1} = lick_stacked{s,j}{t,5};
            end
            if ~isempty(lick_stacked{s,j}{t,3})
                tt = tt + 1;
                lick.first{s,j}(tt,1) = lick_stacked{s,j}{t,3};
            end
            if mod(j,2) == 1
                if ~isempty(lick_stacked{s,j}{t,4})
                    ttt = ttt + 1;
                    lick.outcome{s,(j+1)/2}(ttt,1) = lick_stacked{s,j}{t,4};
                end
                if ~isempty(lick_stacked{s,j}{t,3})
                    lick.stim{s,(j+1)/2}(tt,1) = lick_stacked{s,j}{t,3};
                end
            end
        end
    end
end


data_temp = lick.hist;
target = [1 3 5 7];
data = cell(nSession, numel(target));
for j = 1:numel(target)
    for s = 1:nSession
        data{s,j} = [data_temp{s,target(j)}; data_temp{s,target(j)+1}];
    end
end

binsize = 200;
edges = 1:binsize:8001;
nBins = numel(edges)-1;
All.binary = cell(nSession, numel(target));
All.TrialAve = cell(nSession, numel(target));
All.TrialAve3 = cell(1, numel(target));
for j = 1:numel(target)
    for s = 1:nSession
        nTrial = numel(data{s,j});
        temp = zeros(nTrial, 8000);
        for t = 1:nTrial
            temp2 = data{s,j}{t,1} + 1000;
            temp(t, temp2) = 1;
        end
        All.binary{s,j} = temp;
        All.TrialAve{s,j} = mean(temp,1);
        All.TrialAve3{1,j}(s,:) = All.TrialAve{s,j};
    end
end

timescale = -1:binsize/1000:7-binsize/1000;
All.TrialAveBinned = cell(1, numel(target));
for j = 1:numel(target)
    for s = 1:nSession
        for i = 1:nBins
            All.TrialAveBinned{1,j}(s,i) = (1000/binsize) * sum(All.TrialAve3{1,j}(s,edges(i):edges(i+1)-1));
        end
    end
end

All.TrialAveStat = cell(1, numel(target)/2);
for j = 1:(numel(target)/2)
    for i = 1:nBins
        p = ranksum(All.TrialAveBinned{1,j}(:,i), All.TrialAveBinned{1,j+2}(:,i));
        if isnan(p), p = 1; end
        All.TrialAveStat{1,j}(1,i) = p;
    end
end

figure;
set(gcf,'Position',[200 200 400 400]);
nRow = numel(target);
nPair = nRow/2;

gap = 0.03;
mainH = 0.6/(nPair);
statH = mainH/4;    
blockH = mainH + statH + gap; 
left = 0.15; width = 0.8;

for j = 1:nPair
    top_hist = 1 - (j-1)*blockH;
    ax_hist = axes('Position',[left, top_hist-mainH, width, mainH]);
    bar(timescale, mean(All.TrialAveBinned{1,j},1), 'FaceAlpha', 0.5, 'FaceColor','b'); hold on;
    bar(timescale, mean(All.TrialAveBinned{1,j+2},1), 'FaceAlpha', 0.5, 'FaceColor','r');
    ylim([0 10]);
    if j == 1
        ylabel('Go: Lick rate (Hz)');
    else
        ylabel('No-go: Lick rate (Hz)');
    end
    set(gca,'FontSize',9)
    if j < nPair
        set(gca,'XTickLabel',[]);
    else
        xlabel('Time (s)');
    end

    bottom_stat = top_hist-mainH-statH-gap/2;
    ax_stat = axes('Position',[left, bottom_stat, width, statH]);
    imagesc(timescale, 1, All.TrialAveStat{1,j});
    colormap(ax_stat, gray);
    caxis([0 0.05]);
    set(gca,'YTick',[]);
    if j < nPair
        set(gca,'XTickLabel',[]);
    else
        xlabel('Time (s)');
    end
    ylabel('p val');
end
