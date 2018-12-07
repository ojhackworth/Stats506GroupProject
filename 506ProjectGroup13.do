// 506 Group Project STATA Portion
//
// 506 Project
// Topic: Comparing logistic and probit regression
//
// Author: Olivia Hackworth, ohackwor@umich.edu
// Date Last Modified: 11/26/18


//Load data
clear
import delimited CommunitiesCrimeData.csv


// Logistic Regression

logistic highcrime percapinc pctpopunderpov pcturban population

estat gof
//fail to reject model based on Pearson


 estat gof, group(5) table


// Probit Regression

probit highcrime percapinc pctpopunderpov pcturban population

estat gof
//fail to reject model based on Pearson

estat gof, group(5) table
