
# CountNet3D

CountNet3D é uma arquitetura de visão computacional 3D projetada para inferir a contagem de objetos oclusos em cenas densamente povoadas. Esta implementação usa PyTorch, torch-points3d e RetinaNet para detecção de objetos 2D e PointNet para processamento de nuvens de pontos 3D.

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/seu_usuario/CountNet3D.git
   cd CountNet3D
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado):**

   - No Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - No Linux/macOS:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Instale as dependências:**

   ```bash
   pip install -r requirements.txt
   ```


## Arquitetura

A arquitetura CountNet3D é composta pelos seguintes componentes:

1. **Detecção de Objetos 2D (RetinaNet)**:
   - Utiliza um modelo de detecção de objetos 2D para identificar e localizar objetos em imagens.
   - As detecções são feitas em imagens RGB e as caixas delimitadoras 2D são projetadas para o espaço 3D.

2. **Propostas de PointBeams**:
   - Projeção das caixas delimitadoras 2D detectadas para o espaço 3D usando a pose da câmera, propriedades intrínsecas da câmera e ray casting.
   - Criação de prismas retangulares chamados PointBeams, que se estendem ao longo de um vetor normal ao plano da imagem.

3. **Representação de PointBeams**:
   - **Rotação Ortogonal**: Rotação de cada PointBeam ao redor do eixo vertical para torná-lo ortogonal ao eixo central.
   - **Mean Shift**: Ajuste de cada PointBeam subtraindo a média de todos os pontos na proposta.
   - **Profundidade do Feixe (Beam Depth)**: Cálculo das características de profundidade para cada ponto no feixe.

4. **Backbone PointNet**:
   - Utilizado para aprender características geométricas de cada PointBeam.
   - Consiste em camadas convolucionais e totalmente conectadas para extrair e processar características dos dados de nuvem de pontos.

5. **Dicionário Geométrico**:
   - Mapeamento das classes detectadas para tipos geométricos grosseiros, reduzindo a dimensionalidade do problema.
   - Utilização de uma codificação one-hot para representar o tipo geométrico.

6. **Rede de Estimativa de Contagem**:
   - Combinação das características geométricas do PointNet com a codificação one-hot do tipo geométrico.
   - Passagem dos dados combinados por uma série de camadas totalmente conectadas (MLP) para prever a contagem de objetos em cada PointBeam.

## Referências

Este projeto foi inspirado e baseado no paper:
**CountNet3D: A 3D Computer Vision Approach To Infer Counts of Occluded Objects**

Autores: Porter Jenkins; Kyle Armstrong; Stephen Nelson; Siddhesh Gotad; J. Stockton Jenkins; Wade Wilkey; Tanner Watts

## Créditos

Este código foi desenvolvido com base nas ideias apresentadas no paper "CountNet3D: A 3D Computer Vision Approach To Infer Counts of Occluded Objects". Agradecimentos aos autores do paper pela contribuição à pesquisa em visão computacional.

## Notas:

1. Instalação O README inclui instruções para clonar o repositório, criar um ambiente virtual e instalar as dependências.
2. Uso Descreve como preparar os dados, treinar o modelo e usar o modelo para fazer inferências.
3. Exemplo de Código Inclui um exemplo de código completo para treinar o modelo com dados simulados.
4. Arquitetura Descreve detalhadamente a arquitetura do CountNet3D.
5. Referências e Créditos Dá os devidos créditos ao paper original e aos autores.

Você pode ajustar e expandir o conteúdo conforme necessário para atender às necessidades específicas do seu projeto.