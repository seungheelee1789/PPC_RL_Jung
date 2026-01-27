%% subspace 
clear; clc; close all;
load subspace.mat

nSeg =7; 

for i =1:nSeg
    for j = 1:nSeg
        PPCAC_re(i,j) = mean(PPCAC(i,j,:));
        PPCIC_re(i,j) =mean(PPCIC(i,j,:));
        PPCAC_re2{i,j} = squeeze(PPCAC(i,j,:));
        PPCIC_re2{i,j} = squeeze(PPCIC(i,j,:));
        temp_AC = squeeze(PPCAC(i,j,:));
        temp_IC = squeeze(PPCIC(i,j,:));

        R1_AC = vertcat(squeeze(PPCAC(2,1,:)), squeeze(PPCAC(3,1,:)), squeeze(PPCAC(3,2,:)));
        p_AC = ranksum(R1_AC, squeeze(PPCAC(i,j,:)));
        PPCAC_stat(i,j) = p_AC * 21;

        R1_IC = vertcat(squeeze(PPCIC(2,1,:)), squeeze(PPCIC(3,1,:)), squeeze(PPCIC(3,2,:)));
        p_IC = ranksum(R1_IC, squeeze(PPCIC(i,j,:)));
        PPCIC_stat(i,j) = p_IC * 21;

        p =ranksum(temp_AC, temp_IC);
        AC_IC{1,1}(i,j) = p;
        AC_IC{1,2}(i,j) = PPCAC_re(i,j) - PPCIC_re(i,j);
    end
end
%%
aa = 0.3;
bb = 0.9;
figure()
subplot(1,2,1)
imagesc(PPCAC_re)
colormap(flip(hot))
caxis([aa bb])
axis("square")

subplot(1,2,2)
imagesc(PPCIC_re)
colormap(flip(hot))
caxis([aa bb])
axis("square")
%%
cri = 0.4;
n = 256;
cmap = [linspace(0,1,n/2)', linspace(0,1,n/2)', ones(n/2,1);  % Blue → white
        ones(n/2,1), linspace(1,0,n/2)', linspace(1,0,n/2)']; % White → blue

p_mat = AC_IC{1,1};
m = ((nSeg * nSeg) - nSeg) / 2; 
p_bonf = p_mat * m;    
p_bonf(p_bonf > 1) = 1; 

p_vec = p_mat(:); 
p_fdr = mafdr(p_vec, 'BHFDR', true);
p_fdr_mat  = reshape(p_fdr,  nSeg, nSeg);

p_log = log10(p_bonf + eps);
clims = [log10(1e-6), log10(0.001)]; 

figure()
ax1 = subplot(1,2,1);
imagesc(AC_IC{1,2})
colormap(ax1, cmap)
caxis(ax1, [-cri +cri])
colorbar(ax1)
axis(ax1, 'square')

ax2 = subplot(1,2,2);
imagesc(p_log, clims)
colormap(ax2, gray)
caxis(ax2, clims)
c = colorbar(ax2);
c.Ticks = log10([1e-6 1e-5 1e-4 1e-3 1e-2]);
c.TickLabels = {'1e-6','1e-5','1e-4','1e-3','0.1'};
axis(ax2, 'square')
