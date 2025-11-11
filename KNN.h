//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

#ifndef STACK_PHY_LAYER_MACHINE_LEARNING_KNN_H_
#define STACK_PHY_LAYER_MACHINE_LEARNING_KNN_H_

#include <bits/stdc++.h> // header file for all c++ libraries
#include <fstream>
using namespace std;


struct Vetores_KNN
{
    vector<vector<double>> X_train;
    vector<int> Y_train;
    vector<int> X_dual_train;

    vector<vector<double>> X_test;
    vector<int> Y_test;
    vector<int> X_dual_test;

    vector<vector<double>> threshold_nr;
    vector<vector<double>> threshold_lte;



};

class KNN {
public:
    KNN();
    int Predict(std::vector<double>, int k, struct Vetores_KNN vet);
    vector<vector<string>> load_file();
    struct Vetores_KNN split_dataset(vector<vector<double>> record, struct Vetores_KNN vet);
    int find_max(std::map<int, int> counts);
    double stringParaDouble(const string &str);
    double calcularMedia(const vector<vector<std::string>> &dados, int type_net);
    double calcularDesvioPadrao(const vector<vector<string>> &dados, double media, int type_net);
    vector<vector<double>> normalizarDados(const vector<vector<string>> &dados,double media, double desvioPadrao, int type_net);
    vector<double> normalizarValor(const vector<double> &valor, double media, double desvioPadrao);
    vector<double> desvio_padrao;
    vector<double> media_dataset;
    bool detectarEventoB1(const std::vector<double> &valorNormalizado,
                              double threshold_rsrp, int idx_rsrp,
                              double threshold_sinr, int idx_sinr);
    virtual ~KNN();
private:
    double euclidean_distance(std::vector<double> x, std::vector<double> y);
    double manhatann_distance(std::vector<double> x, std::vector<double> y);
    double chebyshev_distance(std::vector<double> x, std::vector<double> y);

};

#endif /* STACK_PHY_LAYER_MACHINE_LEARNING_KNN_H_ */
