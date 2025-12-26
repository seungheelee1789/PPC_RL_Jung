clear; clc; close all; 
load Fig4_PCA_Hit_CR.mat

data_list = {'PAC_norm','PIC_norm'};
data_title = {'PPCAC', 'PPCIC'}; 
angle_set{1,1} = [-160, -5];
angle_set{1,2} = [-160, 45];

for dd = 1:length(data_list)
    figure('Position',[100+dd*10 100+dd*10 500 200],  'Renderer', 'painters');
    target_data = eval(data_list{dd});
    clear temp datasource

    for s = [9 10]
        temp{1,s} = horzcat(target_data{1,s}, target_data{2,s});
    end

    for s = 1:size(temp,2)
        datasource = horzcat(temp{:});
    end

    clear coeff score latent explained mu
    [coeff,score,latent, ~, explained, mu] = pca(datasource');

    clear datasource3
    curr = 1;
    for j = 1:size(target_data,1)
        for s = 1:size(target_data,2)
            datasource3{1,curr} = target_data{j,s};
            curr = curr + 1;
        end
    end

    cri = 80;
    pc_cri = cri;
    if pc_cri > 3
        pc_num = min(find(cumsum(explained) >= cri))
    elseif pc_cri == 3
        pc_num = 3
    end

    clear scorere
    smth =50;
    for j = 1:size(datasource3,2)
        Xj = datasource3{1,j}';
        Xj = bsxfun(@minus, Xj, mu);
        scorere{j,1} = Xj * coeff;
        scorere{j,2} = smoothdata(scorere{j,1}, 'gaussian', smth);
    end

    timescale=[-pretime/Ca_fps:binsize/1000:(posttime+1)/Ca_fps-binsize/1000];
    wdth = 2;
    sz=2;

    limcri = ceil(max(max(vertcat(scorere{:,1}))));

    subplot(1,2,1)
    datalabelset = {'R1-Hit', 'TR-Hit', 'R2-Hit','R1-CR', 'TR-CR','R2-CR'};
    datalabelset2 = flip(datalabelset);

    colors_set = colormap(winter(9));
    color_mean{1,1} = colors_set(1,:);
    color_mean{1,2} = [155, 48, 255]/255;
    color_mean{1,3} = [18, 10, 143]/255;
    colors_set = colormap(autumn(9));
    color_mean{1,4} = colors_set(1,:);
    color_mean{1,5} = colors_set(6,:);
    color_mean{1,6} = colors_set(8,:);

    target_type = [9 5 10 19 15 20];

    for j = 1:length(target_type)
        i = target_type(j);
        scatter3(scorere{i,2}(:,1),scorere{i,2}(:,2),scorere{i,2}(:,3),sz/4,'MarkerEdgeColor',color_mean{1,j})
        hold on
        grid off
        plot3(scorere{i,2}(:,1),scorere{i,2}(:,2),scorere{i,2}(:,3),'LineWidth', wdth, 'Color',color_mean{1,j})
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
        axis("square")
    end
    view(angle_set{1,dd}(1,1), angle_set{1,dd}(1,2));
    title(data_title{1,dd})




    subplot(1,2,2)
    colors_set = colormap(winter(9));
    for i = [1 2 3 4 6 7 8]
        color_mean{1,i} = colors_set(i+1,:);
    end
    color_mean{1,9} = colors_set(1,:);      % R1 Hit
    color_mean{1,5} = [155, 48, 255]/255;   % TR Hit
    color_mean{1,10} = [18, 10, 143]/255;   % R2 Hit 

    colors_set = colormap(autumn(9));
    for i = [11 12 13 14 16 17 18]
        color_mean{1,i} = colors_set(i-10+1,:);
    end
    color_mean{1,19} = colors_set(1,:); % R1 CR
    color_mean{1,15} = colors_set(6,:); % TR CR
    color_mean{1,20} = colors_set(9,:); % R2 CR

    target_type = [1:20];
    sz = 0.1;

    for j = 1:length(target_type)
        i = target_type(j);
        if ismember(j, [9 10 5 15 19 20])
            wdth = 1;
        else
            wdth = 0.5;
        end
        scatter3(scorere{i,2}(pretime,1),scorere{i,2}(pretime,2),scorere{i,2}(pretime,3),sz,color_mean{1,j},'filled')
        hold on
        grid off
        plot3(scorere{i,2}(:,1),scorere{i,2}(:,2),scorere{i,2}(:,3),'LineWidth', wdth, 'Color',color_mean{1,j})
        xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
        axis("square")
    end
    view(angle_set{1,dd}(1,1), angle_set{1,dd}(1,2));

end