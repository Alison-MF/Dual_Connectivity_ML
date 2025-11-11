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

#include "KNN.h"

KNN::KNN() {
    // TODO Auto-generated constructor stub

}

KNN::~KNN() {
    // TODO Auto-generated destructor stub
}

vector<vector<string>> KNN::load_file()
{
    ifstream file("csv_logistic_dual.csv"); // Carrega o arquivo CSV
    string line;
    // Declaração do vetor record
    vector<vector<string>> record;
    // Faz a leitura do cabeçalho e descarta
    getline(file, line);
    // Loop em todas linhas do arquivo
    while (getline(file, line))
    {
        vector<string> columns; // Vector para armazenae as colunas
        // Criação do objeto stringstream
        stringstream ss(line);
        // While com a função de separar as colunas ao encontrar as vírgulas
        string column;
        while (getline(ss, column, ','))
        {
            columns.push_back(column);
        }
        // Armazena as informações em colunas
        record.push_back(columns);
    }
    file.close();
    return record;
}

struct Vetores_KNN KNN::split_dataset(vector<vector<double>> record,
                                      struct Vetores_KNN vet)
{
    // Realiza um shuffle(embaralha) os elementos dentro do vetor record
    random_shuffle(record.begin(), record.end());
    // Calcula os primeiros 70% do dataset para treinamento
    int num_elements_70 = record.size() * 0.7;
    double mean_rsrp_nr, mean_sinr_nr, mean_distance_nr;
    double mean_rsrp_lte, mean_sinr_lte, mean_distance_lte;
    // vector<vector<double>> X_train;
    //   Separa os elementos nos labels X_train e y_train
    for (int i = 0; i < num_elements_70; i++)
    {
        vector<double> row;
        for (int j = 0; j < record[i].size(); j++)
        {
            if (j < 8)
            {
                row.push_back(record[i][j]);
                if (record[i][8] == 5)
                    vet.threshold_nr.push_back({record[i][5], record[i][6]});
                else
                    vet.threshold_lte.push_back({record[i][5], record[i][6]});
            }
            else if (j == 8)
                vet.X_dual_train.push_back(record[i][j]);

            else if (j == 9)
                vet.Y_train.push_back(record[i][j]);
        }

        vet.X_train.push_back(row);
    }

    // calculo da media das redes

    vector<vector<double>> X_teste;
    // Calcula os primeiros 30% do dataset para teste
    int num_elements_30 = record.size() * 0.3;
    // Adiciona 30% do dataset aos labels de x_test e y_test
    for (int i = num_elements_70; i < (num_elements_70 + num_elements_30);
         i++)
    {
        vector<double> row;
        for (int j = 0; j < record[i].size(); j++)
        {
            if (j < 8)
            {
                row.push_back(record[i][j]);
            }

            else if (j == 8)
                vet.X_dual_test.push_back(record[i][j]);
            else if (j == 9)
                vet.Y_test.push_back(record[i][j]);
        }
        vet.X_test.push_back(row);
    }
    return vet;
}

int KNN::Predict(std::vector<double> val, int k, struct Vetores_KNN vet)
{


    double min = RAND_MAX;
    // std::vector<std::pair<double, std::vector<double>>> Points;
    // vet.Neighbors.clear();
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < vet.X_train.size(); ++i)
    {
        std::vector<double> currentPoint = vet.X_train.at(i);
        double dist = euclidean_distance(val, currentPoint);
        int lbl = vet.Y_train.at(i);

        distances.push_back(std::pair<double, int>(dist, lbl));
        // Points.push_back(
        //       std::pair<double, std::vector<double>>(dist, currentPoint));
    }
    // Sort vectors by distance.
    std::sort(distances.begin(), distances.end());
    // std::sort(Points.begin(), Points.end());

    std::map<int, int> counts;
    double distanceSum = 0.0;

    for (int i = 0; i < k; i++)
    {
        int lbl = distances.at(i).second;
        distanceSum += distances.at(i).first;
        // vet.Neighbors.push_back(
        //       std::pair<std::vector<double>, int>(Points.at(i).second, lbl));

        if (!counts.count(lbl))
            counts.insert(std::pair<int, int>(lbl, 1));
        else
            counts[lbl]++;

        //  std::cout<<"rede "<<type_net <<" distancia "<<distances.at(i).first << "rotulo "<<lbl << " contagem "<< counts[lbl]<<std::endl;
    }
    // std::cout<<"pred "<<find_max(counts)<<std::endl;

    // Calcular a distância média
    double averageDistance = distanceSum / k;
    // std::cout << "Distância média: " << averageDistance << " threshold: " << threshold << std::endl;

    int result = find_max(counts);

   //std::cout << "averageDistance " << averageDistance << " threshold " <<threshold << " result "<<result<<" rede "<<type_net<<std::endl;


    // Aplicar o critério para ajustar a saída se for igual a 1
   /* if (result == 1 && averageDistance > threshold)
    {
        return 1; // Reduzir a classificação
       // std::cout << " ok " << std::endl;
    }*/

    return result;
}


// our distance function
// TODO:Add different distace functions like Manhattan distance. Each distance function has a specific use.
// float Knn::euclideanDistance(std::vector<float> x, std::vector<float> y)
//{
//  return std::sqrt(std::pow(x.at(0) - y.at(0), 2) + std::pow(x.at(1) - y.at(1), 2));
//}

double KNN::euclidean_distance(std::vector<double> x, std::vector<double> y)
{
    // Verificar se os dois vetores têm a mesma dimensão (neste caso, 9 dimensões)
    if (x.size() != 8 || y.size() != 8)
    {
        // Lida com erro de dimensões incorretas
        // Neste exemplo, você pode lançar uma exceção, retornar um valor padrão ou tomar outra ação apropriada
        return -1; // Exemplo de retorno de erro
    }

    // Inicializar a soma dos quadrados das diferenças
    double sumOfSquares = 0.0;

    // Calcular a soma dos quadrados das diferenças de cada dimensão
    for (size_t i = 0; i < 8; ++i)
    {
        sumOfSquares += std::pow(x[i] - y[i], 2);
    }

    // Retornar a raiz quadrada da soma dos quadrados
    return std::sqrt(sumOfSquares);
}

double KNN::manhatann_distance(std::vector<double> x, std::vector<double> y)
{
    // Verificar se os dois vetores têm a mesma dimensão (neste caso, 9 dimensões)
    if (x.size() != 8 || y.size() != 8)
    {
        // Lida com erro de dimensões incorretas
        // Neste exemplo, você pode lançar uma exceção, retornar um valor padrão ou tomar outra ação apropriada
        return -1; // Exemplo de retorno de erro
    }

    // Inicializar a soma dos quadrados das diferenças
    double sum_abs = 0.0;

    // Calcular a soma dos quadrados das diferenças de cada dimensão
    for (size_t i = 0; i < 8; ++i)
    {
        sum_abs += std::abs(x[i] - y[i]);
    }

    // Retornar a raiz quadrada da soma dos quadrados
    return sum_abs;
}

double KNN::chebyshev_distance(std::vector<double> x, std::vector<double> y)
{
    // Verificar se os dois vetores têm a mesma dimensão (neste caso, 8 dimensões)
    if (x.size() != 8 || y.size() != 8)
    {
        // Lida com erro de dimensões incorretas
        return -1; // Exemplo de retorno de erro
    }

    // Inicializar a maior diferença absoluta das coordenadas
    double maxAbsoluteDifference = 0.0;

    // Calcular a maior diferença absoluta das coordenadas de cada dimensão
    for (size_t i = 0; i < 8; ++i)
    {
        double absoluteDifference = std::abs(x[i] - y[i]);
        if (absoluteDifference > maxAbsoluteDifference)
        {
            maxAbsoluteDifference = absoluteDifference;
        }
    }

    // Retornar a maior diferença absoluta
    return maxAbsoluteDifference;
}

// find the actual label by extracting the label from the map with biggest value.
/*int KNN::find_max(std::map<int, int> counts) {
    std::map<int, int>::iterator itr;
    int max = -1;
    int lbl = 0;
    for (itr = counts.begin(); itr != counts.end(); ++itr) {
        if (itr->second > max) {
            max = itr->second;
            lbl = itr->first;
        }
    }
    return lbl;
}*/

int KNN::find_max(std::map<int, int> counts)
{
    std::map<int, int>::iterator itr;
    double max = -1;
    int lbl = -1;
    for (itr = counts.begin(); itr != counts.end(); ++itr)
    {
        if (itr->second > max)
        {
            max = itr->second;
            lbl = itr->first;
        }
    }
    return lbl;
}

// Função para converter string para double
double KNN::stringParaDouble(const string &str)
{
    istringstream iss(str);
    double valor;
    iss >> valor;
    return valor;
}

// Função para calcular a média de um vetor de dados
double KNN::calcularMedia(const vector<vector<std::string>> &dados, int type_net)
{
    double soma = 0.0;
    int count = 0;
    for (const auto &linha : dados)
    {
        if (stoi(linha[9]) == type_net  || type_net==-1)
        {
            for (size_t i = 1; i < linha.size() - 2; ++i)
            { // Ajuste para não incluir a última coluna (rótulos)
                soma += stringParaDouble(linha[i]);
                count++;
            }
        }
    }
    return soma / count;
}

// Função para calcular o desvio padrão de um vetor de dados
double KNN::calcularDesvioPadrao(const vector<vector<string>> &dados,
                                 double media, int type_net)
{
    double somaQuadrados = 0.0;
    int count = 0;
    for (const auto &linha : dados)
    {
        if (stoi(linha[9]) == type_net || type_net==-1) // Ajuste conforme a posição de type_net e Y
        {
            for (size_t i = 1; i < linha.size() - 2; ++i)
            { // Ajuste para não incluir a última coluna (rótulos)
                double valor = stringParaDouble(linha[i]);
                somaQuadrados += (valor - media) * (valor - media);
                count++;
            }
        }
    }
    return sqrt(somaQuadrados / count);
}

// Função para normalizar os dados
vector<vector<double>> KNN::normalizarDados(const vector<vector<string>> &dados,
                                            double media, double desvioPadrao, int type_net)
{

    // Normaliza apenas as features dos dados
    vector<vector<double>> dadosNormalizados;
    for (const auto &linha : dados)
    {
        vector<double> linhaNormalizada;
        if (stoi(linha[9]) == type_net || type_net==-1) // Ajuste conforme a posição de type_net e Y
        {
            for (size_t i = 1; i < linha.size() - 1; ++i) // Ajuste para não incluir a última coluna (saída)
            {
                double valor = stringParaDouble(linha[i]);
                double valorNormalizado;
                if (i < linha.size() - 2)
                    valorNormalizado = (valor - media) / desvioPadrao;
                else
                    valorNormalizado = valor;
                linhaNormalizada.push_back(valorNormalizado);
            }
            // Mantém a última coluna (saída) inalterada
            linhaNormalizada.push_back(stringParaDouble(linha.back())); // Adiciona a última coluna sem normalização
            dadosNormalizados.push_back(linhaNormalizada);
        }
    }
    return dadosNormalizados;
}

vector<double> KNN::normalizarValor(const vector<double> &valor, double media,
                                    double desvioPadrao)
{
    vector<double> valorNormalizado;
    for (size_t i = 0; i < valor.size(); ++i)
    {
        double valorNorm = (valor[i] - media) / desvioPadrao;
        valorNormalizado.push_back(valorNorm);
    }
    return valorNormalizado;
}

bool KNN::detectarEventoB1(const std::vector<double> &valorNormalizado,
                           double threshold_rsrp, int idx_rsrp,
                           double threshold_sinr, int idx_sinr)
{
    return valorNormalizado[idx_rsrp] > threshold_rsrp &&
           valorNormalizado[idx_sinr] > threshold_sinr;
}



