clear
clc
clear
%load 'L1_train.csv'

%% Caricamento matrice
data = readmatrix("L1_train.csv");
matrice = data(35065:end, 1:27);


nTemp = size(matrice, 1);
t = linspace(0, 23, nTemp);


load_data = matrice(:, 2);
matriceTemp = matrice(:, 3:27);
mediaTemp = mean(matriceTemp, 2);



%% Scatter
figure
subplot(3,1,1)
scatter(t, load_data);
xlabel("orario")
ylabel("load")

subplot(3,1,2)
scatter(t, mediaTemp);
xlabel("orario")
ylabel("media tempeperatura")

subplot(3,1,3)
scatter(mediaTemp, load_data);
xlabel("media temperatura")
ylabel("load")




%% Matrice di correlazione e suo grafico
matrice_corr = corr(matriceTemp, load_data);
figure(2)
heatmap(matrice_corr, 'Title', 'Correlazione Temp vs LOAD');



%% Calcolo e scelta modello polinomiale
%load_N = teta_N * mediaTemp ^ N

mediaTemp_2 = [mediaTemp mediaTemp.^2];
mediaTemp_3 = [mediaTemp_2 mediaTemp.^3];
mediaTemp_4 = [mediaTemp_3 mediaTemp.^4];
mediaTemp_5 = [mediaTemp_4 mediaTemp.^5];

%calcolo modello
teta1 = lscov(mediaTemp, load_data);
teta2 = lscov(mediaTemp_2, load_data);
teta3 = lscov(mediaTemp_3, load_data);
teta4 = lscov(mediaTemp_4, load_data);
teta5 = lscov(mediaTemp_5, load_data);


%ssr
ssr1 = (load_data - mediaTemp * teta1)' * (load_data - mediaTemp * teta1);
ssr2 = (load_data - mediaTemp_2 * teta2)' * (load_data - mediaTemp_2 * teta2);
ssr3 = (load_data - mediaTemp_3 * teta3)' * (load_data - mediaTemp_3 * teta3);
ssr4 = (load_data - mediaTemp_4 * teta4)' * (load_data - mediaTemp_4 * teta4);
ssr5 = (load_data - mediaTemp_5 * teta5)' * (load_data - mediaTemp_5 * teta5);