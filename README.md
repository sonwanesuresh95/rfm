
<div align="left">
<img style="width:30%" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/rfm-logos_black.png"><br>
</div>
                                                                                                          
# rfm
rfm: Python Package for RFM Analysis and Customer Segmentation

**rfm** is a Python package that provides **recency, frequency, monetary analysis** results
for a certain transactional dataset within a snap. Its flexible structure and multiple 
functionalities provide easy and intuitive approach to RFM Analysis. It aims to be 
a ready-made python package with high-level and quick prototyping.
On practical hand, **real world data** is easily suited and adapted by the package.
Additionally, it can make colorful, intuitive graphs using a matplotlib backend without 
breaking a sweat.

## Installation
### Dependencies
<ul>
  <li>Python (>=3.7)</li>
  <li>NumPy (>=1.20.1)</li>
  <li>matplotlib (>=3.3.4)</li>
</ul>

To install the current release (Ubuntu and Windows):

```
pip install rfm
```

Example:

```
# predefine a transaction dataset as df

>>> from rfm import RFM

>>> r = RFM(df, customer_id='CustomerID', transaction_date='InvoiceDate', amount='Amount')

>>> r.segment_distribution()
```

<div align="left">
  <img style="height:300px;width:600px" src="https://github.com/sonwanesuresh95/rfm/blob/main/example_/segment_dist.png"><br>
</div>

# Documentation
The official documentation is hosted on :

# License
[MIT](https://github.com/sonwanesuresh95/rfm/blob/main/LICENSE)
