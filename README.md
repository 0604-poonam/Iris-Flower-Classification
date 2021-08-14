# LGMVIP
Task 1

Iris Flower Classification ML Project

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "37bdd90c",
   "metadata": {},
   "source": [
    "# LGMVIP-TASK 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed6ef36f",
   "metadata": {},
   "source": [
    "import required libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "51585c48",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0001c28e",
   "metadata": {},
   "source": [
    "reading the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "9142d66e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(\"C:\\\\Users\\\\House Of Computers\\\\iris.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5bb4265",
   "metadata": {},
   "source": [
    "DATA ANALYSIS"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "defaf378",
   "metadata": {},
   "source": [
    "to represent first five rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "157c4369",
   "metadata": {},
   "outputs": [

{
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>Iris-setosa</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species\n",
       "0   1            5.1           3.5            1.4           0.2  Iris-setosa\n",
       "1   2            4.9           3.0            1.4           0.2  Iris-setosa\n",
       "2   3            4.7           3.2            1.3           0.2  Iris-setosa\n",
       "3   4            4.6           3.1            1.5           0.2  Iris-setosa\n",
       "4   5            5.0           3.6            1.4           0.2  Iris-setosa"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a84d88af",
   "metadata": {},
   "source": [
    "to represent last five rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "654c2fb5",
   "metadata": {},
   "outputs":
{
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>146</td>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>147</td>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>148</td>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>149</td>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>150</td>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>Iris-virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \\\n",
       "145  146            6.7           3.0            5.2           2.3   \n",
       "146  147            6.3           2.5            5.0           1.9   \n",
       "147  148            6.5           3.0            5.2           2.0   \n",
       "148  149            6.2           3.4            5.4           2.3   \n",
       "149  150            5.9           3.0            5.1           1.8   \n",
       "\n",
       "            Species  \n",
       "145  Iris-virginica  \n",
       "146  Iris-virginica  \n",
       "147  Iris-virginica  \n",
       "148  Iris-virginica  \n",
       "149  Iris-virginica  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a622609e",
   "metadata": {},
   "source": [
    "to identify shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "aa5d999d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(150, 6)"
      ]
},
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76d7008b",
   "metadata": {},
   "source": [
    "to identify type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "02d15daa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id                 int64\n",
       "SepalLengthCm    float64\n",
       "SepalWidthCm     float64\n",
       "PetalLengthCm    float64\n",
       "PetalWidthCm     float64\n",
       "Species           object\n",
       "dtype: object"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c5211975",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "      <td>150.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>75.500000</td>\n",
       "      <td>5.843333</td>\n",
       "      <td>3.054000</td>\n",
       "      <td>3.758667</td>\n",
       "      <td>1.198667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>43.445368</td>\n",
       "      <td>0.828066</td>\n",
       "      <td>0.433594</td>\n",
       "      <td>1.764420</td>\n",
       "      <td>0.763161</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>4.300000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>38.250000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>2.800000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>0.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>75.500000</td>\n",
       "      <td>5.800000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>4.350000</td>\n",
       "      <td>1.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>112.750000</td>\n",
       "      <td>6.400000</td>\n",
       "      <td>3.300000</td>\n",
       "      <td>5.100000</td>\n",
       "      <td>1.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>150.000000</td>\n",
       "      <td>7.900000</td>\n",
       "      <td>4.400000</td>\n",
       "      <td>6.900000</td>\n",
       "      <td>2.500000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm\n",
       "count  150.000000     150.000000    150.000000     150.000000    150.000000\n",
       "mean    75.500000       5.843333      3.054000       3.758667      1.198667\n",
       "std     43.445368       0.828066      0.433594       1.764420      0.763161\n",
       "min      1.000000       4.300000      2.000000       1.000000      0.100000\n",
       "25%     38.250000       5.100000      2.800000       1.600000      0.300000\n",
       "50%     75.500000       5.800000      3.000000       4.350000      1.300000\n",
       "75%    112.750000       6.400000      3.300000       5.100000      1.800000\n",
       "max    150.000000       7.900000      4.400000       6.900000      2.500000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "b9e7e344",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150 entries, 0 to 149\n",
      "Data columns (total 6 columns):\n",
      " #   Column         Non-Null Count  Dtype  \n",
      "---  ------         --------------  -----  \n",
      " 0   Id             150 non-null    int64  \n",
      " 1   SepalLengthCm  150 non-null    float64\n",
      " 2   SepalWidthCm   150 non-null    float64\n",
      " 3   PetalLengthCm  150 non-null    float64\n",
      " 4   PetalWidthCm   150 non-null    float64\n",
      " 5   Species        150 non-null    object \n",
      "dtypes: float64(4), int64(1), object(1)\n",
      "memory usage: 7.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9d242546",
   "metadata": {},
   "source": [
    "to find out null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "56e82c41",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Id</th>\n",
       "      <th>SepalLengthCm</th>\n",
       "      <th>SepalWidthCm</th>\n",
       "      <th>PetalLengthCm</th>\n",
       "      <th>PetalWidthCm</th>\n",
       "      <th>Species</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Species\n",
       "0    False          False         False          False         False    False\n",
       "1    False          False         False          False         False    False\n",
       "2    False          False         False          False         False    False\n",
       "3    False          False         False          False         False    False\n",
       "4    False          False         False          False         False    False\n",
       "..     ...            ...           ...            ...           ...      ...\n",
       "145  False          False         False          False         False    False\n",
       "146  False          False         False          False         False    False\n",
       "147  False          False         False          False         False    False\n",
       "148  False          False         False          False         False    False\n",
       "149  False          False         False          False         False    False\n",
       "\n",
       "[150 rows x 6 columns]"
      ]
},
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cedb99e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id               0\n",
       "SepalLengthCm    0\n",
       "SepalWidthCm     0\n",
       "PetalLengthCm    0\n",
       "PetalWidthCm     0\n",
       "Species          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e45352f0",
   "metadata": {},
   "source": [
    "to return the column names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c74959a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',\n",
       "       'Species'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f4a2af2",
   "metadata": {},
   "source": [
    "DATA VISUALIZATION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fffab80",
   "metadata": {},
   "source": [
    "Boxplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1ddb031c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
},
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cedb99e2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Id               0\n",
       "SepalLengthCm    0\n",
       "SepalWidthCm     0\n",
       "PetalLengthCm    0\n",
       "PetalWidthCm     0\n",
       "Species          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e45352f0",
   "metadata": {},
   "source": [
    "to return the column names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "c74959a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',\n",
       "       'Species'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f4a2af2",
   "metadata": {},
   "source": [
    "DATA VISUALIZATION"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1fffab80",
   "metadata": {},
   "source": [
    "Boxplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "1ddb031c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
"source": [
    "sns.pairplot(df.drop('Id',axis=1),hue='Species',data=df,kind='scatter')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b65f3043",
   "metadata": {},
   "source": [
    "Correlation Heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4549d0c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "sns.heatmap(df.corr(),annot=True,cmap=\"YlGnBu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bcc1dec",
   "metadata": {},
   "source": [
    "Distribution Graph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e9383e0d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x720 with 6 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
},
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.hist(edgecolor='black',linewidth=1.5)\n",
    "fig=plt.gcf()\n",
    "fig.set_size_inches(12,10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "84652eb5",
   "metadata": {},
   "source": [
    "Lm Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "beba99e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig=sns.lmplot(x=\"PetalLengthCm\",y=\"PetalWidthCm\",data=df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8cc34f00",
   "metadata": {},
   "source": [
    "Scatter Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b9e51cf5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmEAAAFNCAYAAABIc7ibAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAvoklEQVR4nO3dfbQcd3ng+e8T2ReMrh1rFo8sI4iH8DZgEsmSLSMxjBSxOHIEnGTZxZ4NDN7ZUQhMCANeksxIxthi87LaDCSe4HhDICwZKYS3gxUUh4MtXiSQsS0BBoNDAgRHcoIZxdaVtb7R5dk/qm/cvu6+L327q7q6vp9z+vTt6qquXz3169ajenl+kZlIkiSpXD9SdQMkSZKayCRMkiSpAiZhkiRJFTAJkyRJqoBJmCRJUgVMwiRJkipgEiZJAxAR10XEB6tuh6ThZRImqVIR8eKIOBgRD0XEf4+IAxFxySI/83UR8fkZ094fETsX19onrOf9ETEZEROttn8qIp7Xw+d8JyJe2s+2SRp+JmGSKhMR5wB7gd8F/hnwNOAdwKNVtquTiDijy1u/lZnjwErg74H3l9YoSbVmEiapSs8ByMzdmTmVmacy8y8y8yvTM0TEv4+IeyPiRER8PSIubk3/1Yj4q7bpP9ua/i+Bm4AXtY5Q/UNEbAP+V+BtrWm3tOa9ICI+EhHfj4hvR8Sb2tZ7XUR8OCI+GBEPA6+bbUMy8xHgvwEXdXo/Il4REV9rtWd/q51ExP8LPAO4pdW2t/UWSkl1YxImqUr3AVMR8UcRsSUilrW/GRH/M3Ad8FrgHOAVwA9ab/8V8K+AH6U4evbBiFiRmfcCrwe+kJnjmXluZt4M/DGto1aZ+fKI+BHgFuDLFEfgNgNvjojL25rwSuDDwLmt5buKiHGKRO9wh/eeA+wG3gycB3ySIukay8zXAH8DvLzVtt+aK2iSRoNJmKTKZObDwIuBBP4f4PsR8YmIWN6a5X+nSJy+lIVvZeZ3W8v+aWYezcwfZuafAH8JXLqA1V8CnJeZ12fmZGb+dasNV7bN84XM/HhrHae6fM41EfEPwLeAcTofMXs18GeZ+anM/EdgF3AWsH4B7ZU0Yrpd4yBJpWgduXodQOui9g8C7wKuAp5OccTrCSLitcBbgAtbk8aBpy5g1T8GXNBKoKYtAT7X9vp78/icXZm5fY55LgC+O/0iM38YEd+jOAInqaFMwiQNjcz8RkS8H/iF1qTvAT8+c76I+DGKo1abKY5WTUXEESCmP6rTx894/T3g25n57NmaNP/Wz+oo8MLpFxERFAnm3/Z5PZJqxNORkioTEc+LiLdGxMrW66dTHAH7YmuWP6A43bcmCs9qJWBLKRKX77eWu5rHXxD/d8DKiBibMe2Zba/vAB6OiF+JiLMiYklEXLTY8hhdfAj4mYjYHBFnAm+luAP0YJe2SWoAkzBJVToBrAMORcRJiuTrHookhcz8U+CdFHcdngA+DvyzzPw68H8DX6BIYF4IHGj73NuArwEPRMSDrWnvBZ7fujvx45k5BbwcWAV8G3iQIun70X5vZGZ+E/h5ilIcD7bW+/LMnGzN8uvA9lbbrun3+iUNp8j0KLgkSVLZPBImSZJUAZMwSZKkCpiESZIkVWDgSVjrjqPDEbG3w3sbW4P2Hmk9rh10eyRJkoZBGXXCfhm4l2LIkU4+l5lbS2iHJEnS0BhoEtaq/fMzFLeYv6Ufn/nUpz41L7zwwn581FA5efIkS5curboZQ8e4dGdsOjMu3RmbzoxLZ8alu4XE5q677nowM8/r9N6gj4S9C3gbcPYs87woIr5MUVH6msz82mwfeOGFF3LnnXf2r4VDYv/+/WzcuLHqZgwd49KdsenMuHRnbDozLp0Zl+4WEpuI+G7X9wZVJywitgJXZOYbImIjRYK1dcY85wA/zMyJiLgCeHenIUQiYhuwDWD58uVr9uzZM5A2V2liYoLx8fGqmzF0jEt3xqYz49KdsenMuHRmXLpbSGw2bdp0V2au7fTeIJOwXwdeA5wGnkxxTdhHM/PnZ1nmO8DazHyw2zxr165Nj4Q1h3Hpzth0Zly6MzadGZfOjEt3CzwS1jUJG9jdkZn5a5m5MjMvBK4EbpuZgEXE+a2BbImIS1vt+cGg2iRJkjQsyrg78nEi4vUAmXkT8CrgFyPiNHAKuDIdR0mSJDVAKUlYZu4H9rf+vqlt+o3AjWW0QZIkaZhYMV+SJKkCJmGSJEkVKP2aMEn1MjUF+/bB4cOwejVs2QJLllTdKkmqP5MwSV1NTcHll8OhQ3DyJCxdCuvWwa23mohJ0mJ5OlJSV/v2FQnYxARkFs+HDhXTJUmLYxImqavDh4sjYO1OnoQjRyppjiSNFJMwSV2tXl2cgmy3dCmsWlVJcyRppJiESepqy5biGrDxcYgontetK6ZLkhbHC/MldbVkSXER/r59xSnIVau8O1KS+sUkTNKsliyBrVuLhySpfzwdKUmSVAGTMEmSpAqYhEmSJFXAJEySJKkCJmGSJEkVMAmTJEmqgEmYJElSBUzCJEmSKmASJkmSVAGTMEmSpAqYhEmSJFXAJEySJKkCJmGSJEkVMAmTamZqCvbuhWPHiuepqapbJEnqhUmYVCNTU3D55XDVVXD0aPF8+eUmYpJURyZhUo3s2weHDsHERPF6YqJ4vW9fte2SJC2cSZhUI4cPw8mTj5928iQcOVJJcyRJi2ASJtXI6tWwdOnjpy1dCqtWVdIcSdIimIRJNbJlC6xbB+Pjxevx8eL1li3VtkuStHBnVN0ASfO3ZAncemtxDdiJE7B7d5GALVlSdcskSQvlkTCpZpYsga1bYcWK4tkETJLqySRMkiSpAiZhkiRJFfCaMImi2Om+fUUJiNWrvc5KkjR4JmFqvOkq9IcOFTW3li4t7ji89VYTMUnS4Hg6Uo3XXoU+0yr0kqRymISp8axCL0mqgkmYGs8q9JKkKpiEqfHaq9BHWIVeklQOL8xX47VXoT9ypDgC5t2RkqRBMwmTeKwK/datVbdEktQUno6UJEmqgEmYJElSBQaehEXEkog4HBF7O7wXEfE7EfGtiPhKRFw86PZIkiQNgzKuCftl4F7gnA7vbQGe3XqsA97TepZUcw4FJUmzG2gSFhErgZ8B3gm8pcMsrwQ+kJkJfDEizo2IFZl5bJDtkjRYDgUlSXMb9OnIdwFvA37Y5f2nAd9re31/a5qkGnMoKEmaWxQHoQbwwRFbgSsy8w0RsRG4JjO3zpjnz4Bfz8zPt15/GnhbZt41Y75twDaA5cuXr9mzZ89A2lyliYkJxsfHq27G0DEu3Q1zbI4dg6NHnzj9ggtgxYrBrnuY41I1Y9OZcenMuHS3kNhs2rTprsxc2+m9QZ6O3AC8IiKuAJ4MnBMRH8zMn2+b537g6W2vVwJP+OnOzJuBmwHWrl2bGzduHFijq7J//35GcbsWy7h0N8yx2bsXrruuOAI2bXwcdu+GQTd5mONSNWPTmXHpzLh016/YDOx0ZGb+WmauzMwLgSuB22YkYACfAF7bukvyMuAhrweT6s+hoCRpbqVXzI+I1wNk5k3AJ4ErgG8BjwBXl90eSf3nUFCSNLdSkrDM3A/sb/19U9v0BN5YRhsklcuhoCRpdlbMlyRJqoBJmCRJUgVMwqSKTE7CtdfC5s3F8+Rk1S2SJJWp9AvzJRUJ1/nnw/HjxevbboMbb4QHHoCxsWrbJkkqh0fCpArs3PlYAjbt+PFiuiSpGUzCpAocONB5+sGD5bZDklQdkzCpAhs2dJ6+fn257ZAkVcckTKrA9u2wbNnjpy1bVkyXJDWDSZhUgbGx4iL8HTuKuyN37PCifElqGu+OlCoyNgbXX191KyRJVfFImCRJUgVMwiRJkirg6UiNlFOn4Oqr4Y474NJL4X3vg7POqrpV9TY1Bfv2weHDsHo1bNlSDM4tSf1S9u/MsPyumYRpZJw6BeecA6dPF6+//W34yEfg4YdNxHo1NQWXXw6HDsHJk7B0KaxbB7feaiImqT/K/p0Zpt81T0dqZFx99WMJ2LTTp4vp6s2+fcUP1cQEZBbPhw4V0yWpH8r+nRmm3zWTMI2MO+7oPP1LXyq3HaPk8OHif4rtTp6EI0cqaY6kEVT278ww/a6ZhGlkXHpp5+mXXFJuO0bJ6tXFofp2S5fCqlWVNEfSCCr7d2aYftdMwjQy3vc+OGPGVY5nnFFMV2+2bCmulRgfh4jied26Yrok9UPZvzPD9LvmhfkaGWedVVyEf/XVxSnISy7x7sjFWrKkuFh1377iUP2qVd4dKam/yv6dGabfNZMwjZSzzoI9e6puxWhZsgS2bi0ekjQIZf/ODMvvmqcjJUmSKmASJkmSVAFPR2qk1Knq8rBUbJYkVcMkTCOjTlWXh6lisySpGp6O1MioU9XlYarYLEmqhkmYRkadqi4PU8VmSVI1TMI0MupUdXmYKjZLkqphEqaRUaeqy8NUsVmSVA0vzNfIqFPV5WGq2CxJqoZJmEZKnaouD0vFZklSNTwdKUmSVAGTMEmSpAqYhEmSJFXAa8JqatSHvJnevhMnYO/e0ds+Sc0z6r/bWjiTsBoa9SFv2rfvuuuKxyhtn6TmGfXfbfXG05E1NOpD3rRvH4ze9klqnlH/3VZvTMJqaNSHvBn17ZPUPP6uqROTsBoa9SFvRn37JDWPv2vqxCSshkZ9yJv27YPR2z5JzTPqv9vqjRfm19CoD3nTvn0nTsDu3aO1fZKaZ9R/t9Ubk7CaGvUhb6a3b/9+2Lix6tZI0uKN+u+2Fs7TkZIkSRUYWBIWEU+OiDsi4ssR8bWIeEeHeTZGxEMRcaT1uHZQ7ZEkSRomgzwS9ijwU5n5k8Aq4Kcj4rIO830uM1e1HtcPsD2qkclJuPZauO++4nlycmHLbd68sOWmporK/DfcUDxPTfXe9kGbbuuxY8PfVklSdwO7JiwzE2iV2+TM1iMHtT6NjslJOP98OH4cdu0qEqMbb4QHHoCxsfktB3DbbfNbrk6VrB1NQJJGx0CvCYuIJRFxBPh74FOZeajDbC9qnbLcFxEvGGR7VA87dz6WSE07fryYPojl6lTJ2tEEJGl0RHHAasAriTgX+BjwS5l5T9v0c4AfZuZERFwBvDszn91h+W3ANoDly5ev2bNnz8DbXLaJiQnGpwtjNdx99xWlKQBWrpzg/vuLuJx9NjznOfNbrt1cyx07BkePPnH6BRfAihULaHgJ2tvaHpthbGtV/C51Z2w6My6dGZfuFhKbTZs23ZWZazu+mZmlPIC3A9fMMc93gKfONs+aNWtyFN1+++1VN2Fo7NiRWRyTyty16/Z/+nvHjvkv1/6Ya7lbbskcH3/8MuPjxfRh097W6dgMa1ur4nepO2PTmXHpzLh0t5DYAHdml5xmkHdHntc6AkZEnAW8FPjGjHnOj4ho/X0pxenRHwyqTaqH7dth2bLHT1u2rJg+iOXqVMna0QQkaXTMeWF+RPwc8JvAPwei9cjMPGeORVcAfxQRSyiSqw9l5t6IeD3FB9wEvAr4xYg4DZwCrmxljWqwsbHiYvqdO4tTiTt2FInUbBfXz1zu4EFYv35+y9WpkrWjCUjS6JjP3ZG/Bbw8M+9dyAdn5leA1R2m39T2943AjQv5XDXD2Bhcf31RMX/btoUvt1B1qmTtaAKSNBrmczry7xaagEmSJGl2XY+EtU5DAtwZEX8CfJyiACsAmfnRwTZNkiRpdM12OvLlbX8/Arys7XUCJmEVmpoqrgs6fBhWrx78dUGTk8W1VgcOwIYN87vWqh/rW7myKLo63/X1Gpey4wnlx3TUTe/DEyeKkQQGvQ/r1NckDaeuSVhmXg0QERsy80D7exGxYdANU3dlV3jvtRJ9P9a3kIr5vcalior5Zcd01JU9kkCd+pqk4TWfa8J+d57TVJKyK7z3Wom+7PX1GpcqKuaXHdNRV/ZIAnXqa5KGV9ckLCJeFBFvBc6LiLe0Pa4D/D9bhQ4fLv4X3e7kyaK8wiAcONB5+sGDw7W+XuNSdjyh/JiOurL3YZ36mqThNduRsDFgnOKU5dltj4cp6nupIqtXF6cx2i1dWtS3GoQNXU4+r18/XOvrNS5lxxPKj+moK3sf1qmvSRpeXZOwzPxMZr4DuCwz39H2+O3M/MsS26gZyq7w3msl+rLX12tcqqiYX3ZMR13ZIwnUqa9JGl6zlai4heIuSFojCz1OZr5icM3SbMqu8N5rJfp+rG8hFfN7jUsVFfPLjumoK3skgTr1NUnDa7YSFbtazz8HnA98sPX6KoqBtlWhsiu891qJfrHrW2jF/F7jUkXF/LJjOurKHkmgTn1N0nCarUTFZwAi4obMfEnbW7dExGcH3jJJkqQRNp8SFedFxDOnX0TEvwDOG1yTJEmSRt98BvD+j8D+iPjr1usLgV8YWIskSZIaYM4kLDP/PCKeDTyvNekbmfnobMtIi9XrEDQOCaNh14ThqpqwjVI/zHZ35E9l5m1tA3lP+/GIcABvDUyvQ9A4JIyGXROGq2rCNkr9Mts1Yf+69fzyDg/v69HA9DoEjUPCaNg1YbiqJmyj1C+znY78WETE9EDeUllmG9plttv6e11OKksThqtqwjZK/TLbkbA/AB6MiE9FxHUR8bKIOKeshqm5HBJGo6oJw1U1YRulfplt2KK1wNOBdwKTwJuAv4yIL0fE75XUPjVQr0PQOCSMhl0ThqtqwjZK/TLr3ZGZ+QhFeYovAYeADcBrgZ8uoW1qqF6HoHFIGA27JgxX1YRtlPpltrsj/w2wHlgFPApMJ2IvzswHSmmdGqvXIWgcEkbDrgnDVTVhG6V+mO1I2M3AN4CbgM9m5n3lNEmSJGn0zZaE/SjwkxRHw66LiOcCx4AvAF/IzNtKaJ8kSdJImu3C/KnMvDszb8zMfwNcAewDrgY+VVYD62JqqqjsfsMNxfPU1HCub3ISrr0WNm8unicnB9vOXk1v37Fj5cRTw6Ps71JdLOa7W5fvU6/7frHLlRUX+7Zmmu2asJ+gOAo2/RijOAr2u0CXSjDNVHal9l7XV5dK1r1WzFf9OepBZ4v57tbl+9Trvu/HcmXExb6tTmarE/Z+4AUUR782Z+YzMvPVmfnuzLyzlNbVRNmV2ntdX10qWfdaMV/156gHnS3mu1uX71Ov+74fy8Hw/m5rtM12OvLizPylzNydmd8ts1F1M1ul9mFaX10qWZcdTw0P931ni/nu1iWmvbaz7OV6VZf9oHLNdiRM81R2pfZe11eXStZWvm8u931ni/nu1iWmZY+UUZffbY02k7A+KLtSe6/rq0sl614r5qv+HPWgs8V8d+vyfep13/djORje322Ntlkr5mt+yq7U3uv66lLJuteK+ao/Rz3obDHf3bp8n3rd9/1Yroy42LfVSWRm5zcibgE6vwlk5isG1ajZrF27Nu+8c/TuC9i/fz8bF1IaviGMS3fGpjPj0p2x6cy4dGZcultIbCLirtZ43E8w25GwXT20S5IkSfPQNQnLzM+U2RBJkqQmmfPC/Ih4dkR8OCK+HhF/Pf0oo3Hqri4V+k+dgiuvhGc+s3g+dWqw7ZRGVZ2qrZf9vfd3RnU1nwvz3we8HfgvwCaKYYtikI3S7OpSof/UKTjnHDh9unj97W/DRz4CDz8MZ53V/3ZKo6pO1dbL/t77O6M6m0+JirMy89MUF/F/NzOvA35qsM3SbOpSof/qqx/7YZx2+nQxXdL81anaetnfe39nVGfzScL+v4j4EeAvI+I/RMTPAv98wO3SLOpS6fmOOzpP/9KX+tIsqTHqVG297O+9vzOqs/kkYW8GngK8CVgDvAb4twNsk+ZQl0rPl17aefoll/SlWVJj1Knaetnfe39nVGdzJmGZ+aXMnAAeBt6UmT+XmV8cfNPUTV0q9L/vfXDGjKsOzzijmC5p/upUbb3s772/M6qzOS/Mj4i1FBfnn916/RDwv2XmXQNum7qoS4X+s84qLo69+uri1MAllxQ/jF4sKy1Mnaqtl/2993dGdTafuyP/EHhDZn4OICJeTJGU/cQgG6bZLVkCW7cWj2Fe31lnwZ49g2mT1CRlf+cXo+zvvb8zqqv5XBN2YjoBA8jMzwMnBtckSZKk0TefI2F3RMTvA7spxpJ8NbA/Ii4GyMy7B9g+SZKkkTSfJGxV6/ntM6avp0jKOtYMi4gnA58FntRaz4cz8+0z5gng3cAVwCPA66pO6qamiusuDh8u7kia73UXvS5XF5OTsHMnHDgAGzbA9u0wNjb3couN54kTRXXwUYsnlN9nyt6HVbVzoX2m1/XVyalTxTVTL34x3HTT/K+Z6nXf16Wv9cp/J7ozNguUmQN5UFTVH2/9fSZwCLhsxjxXAPta814GHJrrc9esWZODcvp05ubNmePjmRHF8+bNxfRBLNfu9ttvX1TbB+nRRzOXLcssykQWj2XLiumz6Uc8d+26vad4Druy+0zZ+7BX/WjnQvpMr+urk0ceyTzjjGLbdu26PaF4/cgjsy/X676vS19rt5DvUpX/TpRtof8uGZvOgDuzS04zn7Ejl0fEeyNiX+v18yPi380jucssSltMJ2FnUhw5a/dK4AOteb8InBsRK+b67EHptSp1napZ92LnTjh+/PHTjh8vps+mH/GE0YsnlN9nyt6HVbYT5t/OXtdXJ71WlO9139elr/XKfye6MzYLF0WSNssMRfL1PuA/Z+ZPRsQZwOHMfOGcHx6xBLgLeBbwXzPzV2a8vxf4jSwu9iciPg38SmbeOWO+bcA2gOXLl6/ZM6DbYI4dg6NHnzj9ggtgxSypYa/LtZuYmGB8fHx+M5fsvvuKUzwznX02POc53ZfrRzxXrpzg/vvH57VcnZTdZ8reh73qRzsX0md6XV+d3HMPPPpo8Xd7bJ70JLjoou7L9brv69LX2i3ku1TlvxNlW+i/S8ams02bNt2VmWs7vtntENn0A/hS6/lw27Qjcy034zPOBW4HLpox/c+AF7e9/jSwZrbPGuTpyFtuKQ6Dth9GHx8vpg9iuXbDfDpyx47Hb9v0Y8eO2ZfrRzynT58sNJ7Druw+U/Y+7FU/2rmQPtPr+urk1a9+bLumYwPF9Nn0uu/r0tfaLeS7VOW/E2Vb6L9LxqYzZjkdOZ8Eaj/wPwB3t15fBnxmruU6fM7bgWtmTPt94Kq2198EVsz2OV4TVj6vCes/rwkbXDu9JuzxvCZsbl4T1pnXhHXXryRsPndHvgX4BPDjEXEAOA941VwLRcR5wD9m5j9ExFnAS4HfnDHbJ4D/EBF7gHXAQ5l5bB5tGoheq1LXqZp1L8bG4IEHims6Dh6E9evnd7dTP+J54gTs3j1a8YTy+0zZ+7DKdi6kz/S6vjppryj/pCfBq189v7sje933delrvfLfie6MzcLNeU0YQOs6sOdS3MX4zcz8x3ks8xPAHwFLKIrCfigzr4+I1wNk5k2tEhU3Aj9NUaLi6pxxPdhMa9euzTvvnHWWWtq/fz8bN26suhlDx7h0Z2w6My7dGZvOjEtnxqW7hcQmIrpeE9b1SFhEXAJ8LzMfyMzTEbEG+J+A70bEdZn532dbaWZ+BVjdYfpNbX8n8MZ5bYUkSdIIma1Exe8DkwAR8RLgN4APAA8BNw++aZIkSaNrtmvClrQd7Xo1cHNmfgT4SEQcGXjLJEmSRtisSVhEnJGZp4HNtOp0zWO5RmrskAuqjVEfTqTXYYvKjkud9kNd9v2ocz+MrtmSqd3AZyLiQeAU8DmAiHgWxSlJtUxNweWXFxV+T56EpUth3bribg+/KBoGvfbRuvTt9nZed13xGOT21WW5xajLvh917ofR1vWasMx8J/BW4P0UBVWzbZlfGnzT6qPJQy6oHkZ9OJFehy0qOy512g912fejzv0w2mYdOzIzv5iZH8vMk23T7svMuwfftPo4fLj4H0q7kyeLeifSMOi1j9alb5e9fXVZbjHqsu9HnfthtM05gLfmtnp1cYi43dKlRcE5aRj02kfr0rfL3r66LLcYddn3o879MNpMwvpgy5biHP34OEQUz+vWFdOlYdBrH61L325vJwx+++qy3GLUZd+POvfDaPMuxz5o8pALqodRH06k12GLyo5LnfZDXfb9qHM/jDaTsD5ZsgS2bi0e0jDqtY/WpW9Pt3P/fljISCtlx6VO+6Eu+37UuR9Gl6cjJUmSKmASJkmSVAFPR0oN0WvV7clJ2LkTDhyADRtg+3YYGxvc+spW9vaVvR/a17nQ0QTKVpc+Uxd1imed2tpPJmFSA/RadXtyEs4/H44fL17fdhvceCM88MDsCUBdqnyXvX1l74eZ61zIaAJlq0ufqYs6xbNObe03T0dKDdBr1e2dOx/7h3/a8ePF9EGsr2xlb1/Z+2HmOmF490Vd+kxd1CmedWprv5mESQ3Qa9XtAwc6Tz94cDDrK1vZ21f2fljMOstWl3bWRZ3iWae29ptJmNQAvVbd3rCh8/T16wezvrKVvX1l74fFrLNsdWlnXdQpnnVqa7+ZhEkN0GvV7e3bYdmyx09btqyYPoj1la3s7St7P8xcJwzvvqhLn6mLOsWzTm3tNy/Mlxqg16rbY2PFxd87dxanvtavn99deXWp8l329pW9H2aucyGjCZStLn2mLuoUzzq1td9MwqSG6LXq9tgYXH99eesrW9nbV/Z+aF/nQkcTKFtd+kxd1CmedWprP3k6UpIkqQImYZIkSRUwCZMoigXu3Qs33FA8T01V3aL+K3sbJybgJS8pLiB/yUseq1M1KNPbd+zYcO/DXvdDE/qo1DReE6bGa0K15rK3cWICzj77sdef+1zx+sSJx+7S66dRrwrfhD4qNZFHwtR4TajWXPY2XnHFwqYv1qhXhW9CH5WayCRMjdeEas1lb+NXv9p5+j33DGZ9ddmHZVfalzTcTMLUeE2o1lz2Nr7whZ2nX3TRYNZXl31YdqV9ScPNJEyN14RqzWVv4yc/ubDpizXqVeGb0EelJvLCfDVeE6o1l72N4+PFRfhXXFGcgrzooiIBG8RF+TD6VeGb0EelJjIJk2hGteayt3F8HD772XLWBaNfFb4JfVRqGk9HSpIkVcAkTJIkqQImYZIkSRXwmjCpZqamHrsAfe/ewV+gPb2+w4eLUgnzXV+vy5WtLu2URllTv4cmYVKNlD08z6gPs1OXdkqjrMnfQ09HSjVS9vA8oz7MTl3aKY2yJn8PTcKkGil7+JpRH2anLu2URlmTv4cmYVKNlD18zagPs1OXdkqjrMnfQ5MwqUbKHp5n1IfZqUs7pVHW5O+hF+ZLNVL28DyjPsxOXdopjbImfw9NwqSaKXt4nlEfZqcu7ZRGWVO/h56OlCRJqsDAkrCIeHpE3B4R90bE1yLilzvMszEiHoqII63HtYNqjyRJ0jAZ5OnI08BbM/PuiDgbuCsiPpWZX58x3+cys2EHIJuj7CrITai63GvF/FGvfF8Xk5OwcyccOAAbNsD27TA2VnWr+ss+I83PwJKwzDwGHGv9fSIi7gWeBsxMwjSiyq6C3ISqy71WzB/1yvd1MTkJ558Px48Xr2+7DW68ER54YHQSMfuMNH+lXBMWERcCq4FDHd5+UUR8OSL2RcQLymiPylF2FeQmVF3utWL+qFe+r4udOx9LwKYdP15MHxX2GWn+IjMHu4KIceAzwDsz86Mz3jsH+GFmTkTEFcC7M/PZHT5jG7ANYPny5Wv27Nkz0DZXYWJigvHp4k8j4tgxOHr0idMvuABWrJjfZywkLv1Y37Br38aVKye4//4iNnNtY6+xqWNMh/m7dN99xWnkmc4+G57znMGvv4zY2GdGh3HpbiGx2bRp012Zubbjm5k5sAdwJnAr8JZ5zv8d4KmzzbNmzZocRbfffnvVTei7W27JHB/PLP4/XDzGx4vp87WQuPRjfcOufRt37bp93tvYa2zqGNNh/i7t2PH4WE4/duwoZ/1lxMY+MzqMS3cLiQ1wZ3bJaQZ5d2QA7wXuzczf7jLP+a35iIhLKU6P/mBQbVK5yq6C3ISqy71WzB/1yvd1sX07LFv2+GnLlhXTR4V9Rpq/Qd4duQF4DfDViDjSmvafgGcAZOZNwKuAX4yI08Ap4MpW1qgRUHYV5CZUXe61Yv6oV76vi7Gx4iL8nTvh4EFYv3707o60z0jzN8i7Iz8PxBzz3AjcOKg2qHplV0FuQtXlXivmj3rl+7oYG4Prr6+6FYNln5Hmx4r5kiRJFTAJkyRJqoBJmAZqaqqo6n7DDcXz1FTVLaq/yUm49tqi3MG11xavJUn1M8gL89VwVs7uv/aK67t2FcntqFVcl6Sm8EiYBsbK2f3XhIrrktQUJmEamMOHiyNg7U6eLG5bV28OHOg8/eDBctshSVo8kzANzOrVxSnIdkuXFnWD1JsNGzpPX7++3HZIkhbPJEwDY+Xs/mtCxXVJagovzNfAWDm7/9orrp99NuzYMXoV1yWpKUzCNFBWzu6/6Yrr+/fDtm1Vt0aS1CtPR0qSJFXAJEySJKkCJmGSJEkVMAmr2PSwPseOOaxPO+PSnbGpN4fykjTNJKxC08P6XHUVHD1aPF9+uT/KxqU7Y1Nv7fvv7W93/0lNZxJWofZhfcBhfaYZl+6MTb05lJekdiZhFXJYn86MS3fGpt7cf5LamYRVyGF9OjMu3RmbenP/SWpnElah9mF9wGF9phmX7oxNvTmUl6R2VsyvUPuwPidOwO7dDusDxmU2xqbeHMpLUjuTsIpND+uzfz9s3Fh1a4aHcenO2NSbQ3lJmubpSEmSpAqYhEmSJFXAJEzSSJichGuvhfvuK54nJwe7PivfS1osrwmTVHuTk3D++XD8OOzaVSRGN94IDzwAY2P9X9905ftDh4o6X0uXFnc53nqrF9lLmj+PhEmqvZ07iwSs3fHjxfRBsPK9pH4wCZNUewcOdJ5+8OBg1mfle0n9YBImqfY2bOg8ff36wazPyveS+sEkTFLtbd8Oy5Y9ftqyZcX0QbDyvaR+8MJ8SbU3NlZchL9zJ5x9NuzYUSRgg7goH6x8L6k/TMIkjYSxMbj++mIkgW3bBr8+K99LWixPR0qSJFXAJEySJKkCJmGSJEkVMAmTJEmqgEmYJElSBUzCJEmSKmASJkmSVAGTMEmSpAqYhEmSJFXAJEySJKkCJmGSJEkVMAmTJEmqwMCSsIh4ekTcHhH3RsTXIuKXO8wTEfE7EfGtiPhKRFw8qPaoXqamYO9eOHaseJ6aqrpFkiT11yCPhJ0G3pqZ/xK4DHhjRDx/xjxbgGe3HtuA9wywPaqJqSm4/HK46io4erR4vvxyEzFJ0mgZWBKWmccy8+7W3yeAe4GnzZjtlcAHsvBF4NyIWDGoNqke9u2DQ4dgYqJ4PTFRvN63r9p2SZLUT5GZg19JxIXAZ4GLMvPhtul7gd/IzM+3Xn8a+JXMvHPG8tsojpSxfPnyNXv27Bl4m8s2MTHB+Ph41c0YCseOFUfAAFaunOD++4u4XHABrDBF/yf2mc6MS3fGpjPj0plx6W4hsdm0adNdmbm203tn9LVVHUTEOPAR4M3tCdj02x0WeUJWmJk3AzcDrF27Njdu3NjvZlZu//79jOJ29WLvXrjuuuII2K5d+7nmmo2Mj8Pu3WCIHmOf6cy4dGdsOjMunRmX7voVm4HeHRkRZ1IkYH+cmR/tMMv9wNPbXq8Ejg6yTRp+W7bAunUw/Z+M8fHi9ZYt1bZLkqR+GuTdkQG8F7g3M3+7y2yfAF7bukvyMuChzDw2qDapHpYsgVtvLY58XXBB8XzrrcV0SZJGxSBPR24AXgN8NSKOtKb9J+AZAJl5E/BJ4ArgW8AjwNUDbI9qZMkS2LoV9u/3FKQkaTQNLAlrXWzf6Zqv9nkSeOOg2iBJkjSsrJgvSZJUAZMwSZKkCpiESZIkVcAkTJIkqQImYZIkSRUwCZMkSaqASZgkSVIFShnAu58i4vvAd6tuxwA8FXiw6kYMIePSnbHpzLh0Z2w6My6dGZfuFhKbH8vM8zq9UbskbFRFxJ3dRllvMuPSnbHpzLh0Z2w6My6dGZfu+hUbT0dKkiRVwCRMkiSpAiZhw+PmqhswpIxLd8amM+PSnbHpzLh0Zly660tsvCZMkiSpAh4JkyRJqoBJWMkiYklEHI6IvR3e2xgRD0XEkdbj2iraWIWI+E5EfLW13Xd2eD8i4nci4lsR8ZWIuLiKdpZtHnFpcp85NyI+HBHfiIh7I+JFM95vap+ZKy6N7DMR8dy2bT4SEQ9HxJtnzNO4PjPPuDSyzwBExH+MiK9FxD0RsTsinjzj/UX1mTP621zNwy8D9wLndHn/c5m5tcT2DJNNmdmt7soW4NmtxzrgPa3nJpgtLtDcPvNu4M8z81URMQY8Zcb7Te0zc8UFGthnMvObwCoo/jMM/C3wsRmzNa7PzDMu0MA+ExFPA94EPD8zT0XEh4Argfe3zbaoPuORsBJFxErgZ4A/qLotNfRK4ANZ+CJwbkSsqLpRqkZEnAO8BHgvQGZOZuY/zJitcX1mnnERbAb+KjNnFv5uXJ+ZoVtcmuwM4KyIOIPiPzRHZ7y/qD5jElaudwFvA344yzwviogvR8S+iHhBOc0aCgn8RUTcFRHbOrz/NOB7ba/vb00bdXPFBZrZZ54JfB94X+v0/h9ExNIZ8zSxz8wnLtDMPtPuSmB3h+lN7DPtusUFGthnMvNvgV3A3wDHgIcy8y9mzLaoPmMSVpKI2Ar8fWbeNctsd1MMb/CTwO8CHy+jbUNiQ2ZeTHFo940R8ZIZ70eHZZpwa+9ccWlqnzkDuBh4T2auBk4Cvzpjnib2mfnEpal9BoDWKdpXAH/a6e0O00a9zwBzxqWRfSYillEc6foXwAXA0oj4+ZmzdVh03n3GJKw8G4BXRMR3gD3AT0XEB9tnyMyHM3Oi9fcngTMj4qmlt7QCmXm09fz3FNcjXDpjlvuBp7e9XskTDwuPnLni0uA+cz9wf2Year3+MEXyMXOepvWZOePS4D4zbQtwd2b+XYf3mthnpnWNS4P7zEuBb2fm9zPzH4GPAutnzLOoPmMSVpLM/LXMXJmZF1Ic8r0tMx+XUUfE+RERrb8vpdg/Pyi9sSWLiKURcfb038DLgHtmzPYJ4LWtO1EuozgsfKzkppZqPnFpap/JzAeA70XEc1uTNgNfnzFb4/rMfOLS1D7T5iq6n3JrXJ9p0zUuDe4zfwNcFhFPaW3/Zoob69otqs94d2TFIuL1AJl5E/Aq4Bcj4jRwCrgym1FNdznwsdZ3/Azgv2Xmn8+IzSeBK4BvAY8AV1fU1jLNJy5N7TMAvwT8ces0yl8DV9tngLnj0tg+ExFPAf5H4BfapjW+z8wjLo3sM5l5KCI+THE69jRwGLi5n33GivmSJEkV8HSkJElSBUzCJEmSKmASJkmSVAGTMEmSpAqYhEmSJFXAJEzS0IiI/xwRX4uIr0TEkYjo6+DJEbExIvbOd3of13tuRLyhrPVJqgfrhEkaChHxImArcHFmPtqqyD1WcbP65VzgDcDvVdwOSUPEI2GShsUK4MHMfBQgMx+cHrYpItZExGdaA5nfGhErWtP3R8S7IuJgRNzTquZNRFzamna49fzcrmudRUS8LCK+EBF3R8SfRsR4a/p3IuIdrelfjYjntaafFxGfak3//Yj4biuZ/A3gx1tH9/6v1sePR8SHI+IbEfHH0xXJJTWHSZikYfEXwNMj4r6I+L2I+NcAEXEmxaDBr8rMNcAfAu9sW25pZq6nONL0h61p3wBe0hrE+lrg/1xoY1rJ03bgpa1B1O8E3tI2y4Ot6e8BrmlNezvFkGQXU4z1+YzW9F8F/iozV2Xm/9Gathp4M/B84JkU48tKahBPR0oaCpk5ERFrgH8FbAL+JCJ+lSL5uQj4VOtg0RKgfWy23a3lPxsR50TEucDZwB9FxLOBBM7soUmXUSRIB1rrHQO+0Pb+R1vPdwE/1/r7xcDPttrz5xFxfJbPvyMz7weIiCPAhcDne2inpJoyCZM0NDJzCtgP7I+IrwL/liLJ+VpmvqjbYh1e3wDcnpk/GxEXtj5zoQL4VGZe1eX9R1vPUzz2W7qQU4qPtv3d/hmSGsLTkZKGQkQ8t3Xkatoq4LvAN4HzWhfuExFnRsQL2uZ7dWv6i4GHMvMh4EeBv229/7oem/RFYENEPKv1+U+JiOfMsczngf+lNf/LgGWt6Scojs5J0j8xCZM0LMYpTiF+PSK+QnEq8LrMnAReBfxmRHwZOAKsb1vueEQcBG4C/l1r2m8Bvx4RByhOX87H5oi4f/oBPIsigdvdas8XgefN8RnvAF4WEXcDWyhOm57IzB9QnNa8p+3CfEkNF5kzj+RLUj1ExH7gmsy8s+q2AETEk4CpzDzdOnL3nsxcVXGzJA0pr0GQpP55BvChiPgRYBL49xW3R9IQ80iYJElSBbwmTJIkqQImYZIkSRUwCZMkSaqASZgkSVIFTMIkSZIqYBImSZJUgf8fsGQij9F4Mc8AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x=df[\"SepalLengthCm\"]\n",
    "y=df[\"SepalWidthCm\"]\n",
    "plt.figure(figsize=(10,5))\n",
    "plt.title(\"Scatter Plot\")\n",
    "plt.xlabel(\"Sepal Length\")\n",
    "plt.ylabel(\"Sepal Width\")\n",
    "plt.scatter(x,y,marker='.',c='b',s=100)\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8585787",
   "metadata": {},
   "source": [
    "Violin Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "e7363b5a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }

],
   "source": [
    "x=(df[\"SepalLengthCm\"])\n",
    "a=(df[\"SepalWidthCm\"])\n",
    "plt.figure(figsize=(5,4))\n",
    "data=list([df[\"SepalLengthCm\"],df[\"SepalWidthCm\"]])\n",
    "plt.title(\"Violinplot\")\n",
    "plt.xlabel(\"Sepal Length\")\n",
    "plt.ylabel(\"Sepal Width\")\n",
    "plt.violinplot(data)\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73671b8e",
   "metadata": {},
   "source": [
    "Pie Chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "c79ee440",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "a=plt.subplots(1,1,figsize=(10,8))\n",
    "df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(5,5))\n",
    "plt.title(\"Iris Species %\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f9a9bde2",
   "metadata": {},
   "source": [
    "Doughnut Chart"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d0f953d2",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.tit=plt.subplots(1,1,figsize=(10,8))\n",
    "df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8),radius=0.80)\n",
    "plt.title(\"Iris Species %\")\n",
    "plt.pie([1],colors=['w'],radius=0.40)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6e880411",
   "metadata": {},
   "source": [
    "SPLITTING OF DATASET"
   ]
},
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "c8a32ad7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=df.drop('Id',axis=1)\n",
    "X=df.iloc[:,:-1].values\n",
    "y=df.iloc[:,4].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "8a5b6415",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "11a65cd7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "90\n"
     ]
    }
   ],
   "source": [
    "print(len(X_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "66c2a5b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "60\n"
     ]
    }
   ],
   "source": [
    "print(len(X_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf3355a5",
   "metadata": {},
   "source": [
    "TRANING THE MODEL"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bd9a809",
   "metadata": {},
   "source": [
    "KNeighbors Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "3c4eecab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9833333333333333\n"
     ]
    }
   ],
   "source": [
    "model=KNeighborsClassifier()\n",
    "model.fit(X_train,y_train)\n",
    "predictions=model.predict(X_test)\n",
    "print(accuracy_score(y_test,predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "abcff6fa",
   "metadata": {},
   "source": [
    "Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "965cc7c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9833333333333333\n"
     ]
    }
   ],
   "source": [
    "model=RandomForestClassifier(n_estimators=5)\n",
    "model.fit(X_train,y_train)\n",
    "predictions=model.predict(X_test)\n",
    "print(accuracy_score(y_test,predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac70323a",
   "metadata": {},
   "source": [
    "Support Vector Machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "e85b1ebc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0\n"
     ]
    }
   ],
   "source": [
    "model=SVC()\n",
    "model.fit(X_train,y_train)\n",
    "predictions=model.predict(X_test)\n",
    "print(accuracy_score(y_test,predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c516665",
   "metadata": {},
   "source": [
    "Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "cbe6b879",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0\n"
     ]
    }
   ],
   "source": [
    "model=LogisticRegression()\n",
    "model.fit(X_train,y_train)\n",
    "predictions=model.predict(X_test)\n",
    "print(accuracy_score(y_test,predictions))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6fef04d6",
   "metadata": {},
   "source": [
"Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "18530c5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[29,  0,  0],\n",
       "       [ 0, 23,  0],\n",
       "       [ 0,  0, 23]], dtype=int64)"
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import metrics\n",
    "cnf_matrix = metrics.confusion_matrix(y_test,predictions)\n",
    "cnf_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ee9bf0d",
   "metadata": {},
   "source": [
    "Confusision Matrix Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "4158f8a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 257.44, 'Predicted label')"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "class_names=[0,1] \n",
    "fig, ax = plt.subplots()\n",
    "tick_marks = np.arange(len(class_names))\n",
    "plt.xticks(tick_marks, class_names)\n",
    "plt.yticks(tick_marks, class_names)\n",
    "\n",
    "sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap=\"YlGnBu\" ,fmt='g')\n",
    "ax.xaxis.set_label_position(\"top\")\n",
    "plt.tight_layout()\n",
    "plt.title('Confusion matrix', y=1.1)\n",
    "plt.ylabel('Actual label')\n",
    "plt.xlabel('Predicted label')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "64160dd6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

