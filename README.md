# PredictDeliveryType

Predicting vaginal delivery after labour induction using machine learning: development of a multivariable prediction model

## Authors

- Iolanda Ferreira, Obstetrics Department, Centro Hospitalar e Universitário de Coimbra; University of Coimbra, Medicine Faculty - Coimbra, Portugal
- Joana Simões, University of Coimbra, Centre for Informatics and Systems of the University of Coimbra, Department of Informatics Engineering, Portugal
- João Correia, University of Coimbra, Centre for Informatics and Systems of the University of Coimbra, Department of Informatics Engineering, Portugal
- Ana Luísa Areia, Obstetrics Department, Centro Hospitalar e Universitário de Coimbra; University of Coimbra, Medicine Faculty - Coimbra, Portugal

## Packages versions

### Python

- Python: version 3.8.15

### Libraries

The requirements.txt file contains all the version of the libraries used in this program. Install the packages with the command:
        pip install -r requirements.txt

To run the program, it is highly recommended to use the these versions.

## Run Program

To use the program, the following command should be run in the terminal:
        cd src
        python main.py config.json

The *config.json* file contains all the information needed to test different parameters. His content in described in the subsection bellow.

### Configuration file

It should be given to the program a json file with the following parameters:

<table>

<tr>
    <th> Parameter </th>
    <th> Description </th>
    <th> Available options </th>
    <th> Mandatory </th>
    <th> Default </th>
    <th> Data type </th>
</tr>

<tr>
    <td> dataset </td>
    <td> Path to the dataset </td>
    <td> - </td>
    <td> - </td>
    <td> Yes </td>
    <td> String </td>
</tr>


<tr>
    <td> norm_model </td>
    <td> Normalization / standarization method </td>
    <td> 'MinMax' (MinMaxScaler), 'Robust' (RobustScaler), 'StandardScaler' </td>
    <td> StandardScaler </td>
    <td> No </td>
    <td> String </td>
</tr>

<tr>
    <td> models_names </td>
    <td> List with the models and parameters to use for each model. In case of grid search, each parameter should be a list with the values to test</td>
    <td> Example usage: [[{model's name}* , {dict with the parameters}]] </td>
    <td> - </td>
    <td> Yes </td>
    <td> list of list </td>
</tr>

<tr>
    <td> target_colum </td>
    <td> Name of the column to predict</td>
    <td> - </td>
    <td> Yes </td>
    <td> - </td>
    <td> String </td>
</tr>
<tr>
    <td> numeric_features </td>
    <td> List with the numeric features to scale </td>
    <td> - </td>
    <td> No </td>
    <td> - </td>
    <td> List of strings </td>
</tr>

<tr>
    <td> test_shap </td>
    <td> To test th feature importance with SHAP values. The model_names list should only contain one model</td>
    <td> - </td>
    <td> No </td>
    <td> False </td>
    <td> boolean </td>
</tr>

The config.json present an example of a configuration file.

</table>

\* used in sklearn
