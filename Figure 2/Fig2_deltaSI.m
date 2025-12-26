clear;clc; close all
load Fig2_deltaSI.mat 

delta_SI = SI_delta_stim; % SI_delta_stim for Figure 2M / SI_delta_outcome for Figure 2N 
area = {'AC','IC'};
phase = {'TR','R2'};
nArea = numel(area);
curr = 1;
figure()
set(gcf,'Position',[200 200 350 200*nArea]);
for dd = 1:nArea
    for s = 1:numel(phase)
        subplot(nArea, 2, curr)
        hold on
        violinplot(delta_SI{s,dd}, [], 'ShowData', false, 'ShowMedian', true, 'ViolinAlpha', 0.5, 'MarkerSize', 10);
        mstr = sprintf('%.3f', median(delta_SI{s,dd}));
        title(sprintf('%s, %s, median = %s',area{dd}, phase{s}, mstr));
        p = signrank(delta_SI{s,dd},0);

        [f,xi] = ksdensity(delta_SI{s,dd}); 
        [~,im] = max(f);
        y_mode = xi(im);
        yline(y_mode,'--','LabelHorizontalAlignment','center');

        pstr = sprintf('%.3f', p);
        ylabel(sprintf('p=%s', pstr));
        box on
        curr = curr + 1;
        ylim([-1 1])
    end
end


for dd = 1:nArea
    x = delta_SI{1,dd}; % deltaSI = |TR| - |R1|
    y = delta_SI{2,dd}; % deltaSI = |R2| - |R1|
    temp = [x', y'];
    group_label = [ones(1,length(x')), 2*ones(1,length(y'))];

    all_median = median(temp);
    is_above = temp > all_median;

    try
        [~, ~, p_mood] = crosstab(group_label, is_above);
    catch ME
        p_mood = NaN;
    end
    delta_SI{3,dd} = p_mood;
end
