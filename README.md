<br>
<br>
<div align="left">
<img style="width:400px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm-logos_black.png"><br>
</div>
<br>

[![PyPI Latest Release](https://img.shields.io/pypi/v/rfm.svg)](https://pypi.org/project/rfm/)
![PyPI](https://badge.fury.io/py/rfm.svg)
[![Package Status](https://img.shields.io/pypi/status/rfm.svg)](https://pypi.org/project/rfm/)
[![License](https://img.shields.io/pypi/l/rfm.svg)](https://github.com/sonwanesuresh95/rfm/blob/main/LICENSE)
![Downloads Per Month](https://img.shields.io/pypi/dm/rfm)

# rfm
<b>rfm: Python Package for RFM Analysis and Customer Segmentation</b>

## Info

**rfm** is a Python package that provides **recency, frequency, monetary analysis** results
for a certain transactional dataset within a snap. Its flexible structure and multiple automated
functionalities provide easy and intuitive approach to RFM Analysis in an automated fashion.
It aims to be a ready-made python package with high-level and quick prototyping.
On practical hand, **real world data** is easily suited and adapted by the package.
Additionally, it can make colorful, intuitive graphs using a matplotlib backend without 
breaking a sweat.

## Installation
### Dependencies
<ul>
  <li>Python (>=3.7)</li>
  <li>Pandas (>=1.2.4)</li>
  <li>NumPy (>=1.20.1)</li>
  <li>matplotlib (>=3.3.4)</li>
</ul>

To install the current release (Ubuntu and Windows):

```
$ pip install rfm
```

## Usage

```
# predefine a transaction dataset as df

>>> from rfm import RFM

>>> r = RFM(df, customer_id='CustomerID', transaction_date='InvoiceDate', amount='Amount')

>>> r.segment_distribution()
```

<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_seg_dist.png"><br>
</div>


# License
[MIT](https://github.com/sonwanesuresh95/rfm/blob/main/LICENSE)

# Documentation
<-- Temporarily Hosted Here -->
## Initialization
Read required dataframe
```
>>> df = pd.read_csv('~./data.csv')
```

Import RFM package and start rfm analysis automatically:
```
>>> from rfm import RFM

>>> rfm = RFM(df, customer_id='CustomerID', transaction_date='InvoiceDate', amount='Amount') 

>>> rfm.rfm_table
```
If you want to do rfm analysis manually:
```
>>> rfm.RFM(df, customer_id='CustomerID', transaction_date='InvoiceDate', amount='Amount', automated=False)
```

## Attributes

### RFM.rfm_table
returns resultant rfm table df generated with recency, frequency & monetary values and scores along with segments
```
>>> rfm.rfm_table
```
<div align="left">
  <img style="width:500px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_table.png"><br>
</div>

### RFM.segment_table
returns segment table df with 10 unique categories i.e. Champions, Loyal Accounts etc. 
```
>>> rfm.segment_table
```
<div align="left">
  <img style="height:250px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_segment_table.png"><br>
</div>

## Methods

### RFM.plot_rfm_histograms()
Plots recency, frequency and monetary histograms in a single row
```
>>> rfm.plot_rfm_histograms()
```
<div align="left">
  <img style="width:700px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_histograms.png"><br>
</div>

### RFM.plot_rfm_order_distribution()
Plots orders by customer number
```
>>> rfm.plot_rfm_order_distribution()
```
<div align="left">
  <img style="width:700px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_order_dist.png"><br>
</div>

### RFM.plot_versace_plot(column1, column2)
Plots scatterplot of two input columns

```
>>> rfm.plot_versace_plot(column1='recency',column2='monetary_value')
```

<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_rm.png"><br>
</div>

```
>>> rfm.plot_versace_plot(column1='recency',column2='frequency')
```

<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_rf.png"><br>
</div>

```
>>> rfm.plot_versace_plot(column1='frequency',column2='monetary_value')
```

<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_fm.png"><br>
</div>

### RFM.plot_distribution_by_segment(column, take)
Plots Distribution of input column by segment
```
>>> rfm.plot_distribution_by_segment(column='recency',take='median')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_mrdian_rec.png"><br>
</div>

```
>>> rfm.plot_distribution_by_segment(column='frequency',take='median')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_median_freq.png"><br>
</div>

```
>>> rfm.plot_distribution_by_segment(column='monetary_value',take='median')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_med_mon.png"><br>
</div>

### RFM.plot_column_distribution(column)
Plots column distribution of input column
```
>>> rfm.plot_column_distribution(column='recency')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_col_dist_rec.png"><br>
</div>

```
>>> rfm.plot_column_distribution(column='frequency')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_col_dist_freq.png"><br>
</div>

```
>>> rfm.plot_column_distribution(column='monetary_value')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_col_dist_mon.png"><br>
</div>

### RFM.plot_segment_distribution()
```
>>> rfm.plot_segment_distribution()
```
Plots Segment Distribution, i.e. Segments vs no. of customers
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_seg_dist.png"><br>
</div>

### RFM.find_customers(segment)
returns rfm results df with input category
```
>>> rfm.find_customers('Champions')
```
<div align="left">
  <img style="width:550px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm_champions.png"><br>
</div>

