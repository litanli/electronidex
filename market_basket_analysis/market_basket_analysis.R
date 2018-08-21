#################
# Project Summary
#################

# Datasets:    blackwell_transactions2017.csv (transactions), rules.csv 
#              (discovered association rules)
#
# Scenario:    Electronidex is a mock online and brick-and-mortar electronics 
#              retailer who's considering acquiring a startup retailer called 
#              Blackwell Electronics. They've given us one month's worth of 
#              their transaction data. We want to discover their customer buying 
#              patterns in order to aid upper management in deciding whether to 
#              acquire.
#
# Goal:        Perform a market basket analysis on Blackwell's provided 
#              transaction data. To do this, we will use the Aprior algorithm to 
#              discover association rules, and sort out the actionable rules 
#              from the obvious and inexplicable ones.
#
# Conclusion:  see market_basket_analysis_report document for discussions and
#              conclusions



###############
# Housekeeping
###############

rm(list = ls())
setwd("C:/Users/Litan Li/Desktop/electronidex/market_basket_analysis")

################
# Load packages
################

library(arules)
library(arulesViz)
library(ggplot2)


#####################
# Parallel processing
#####################

library(doParallel) 

# Check number of cores and workers available 
detectCores()
cl <- makeCluster(detectCores()-1, type='PSOCK')
# Start parallel cluster
registerDoParallel(cl)


#############
# Import data
#############

# Load transaction data into a sparse matrix. Columns are items, rows are 
# transactions. Columns are sorted alphabetically.
transactions <- read.transactions("blackwell_transactions2017.csv", 
    format = "basket", sep = ",")

################
# Evaluate data
################

summary(transactions)
# transactions are represented as a sparse matrix with 9835 rows (transactions/
# itemsets) and 125 columns (items) with a density of 0.035.

# most frequently bought items:
# 1. iMac 2519
# 2. HP Laptop 1909
# 3. CYBERPOWER Gamer Desktop 1809
# 4. Apple Earpods 1715
# 5. Apple MacBook Air 1530


# transaction size distribution:
# number of items/transaction    0    1    2    3    4    5    6    7    8    9
# frequency                      2 2163 1647 1294 1021  856  646  540  439  353
#    10   11   12   13  14  15  16   17   18   19   20   21   22   23   25   26    
#   247  171  119   77  72  56  41   26   20   10   10   10    5    3    1    1 
#    27   29   30
#     3    1    1
size_dist <- data.frame(transaction_size = c(0:30), 
                        frequency = c(2, 2163, 1647, 1294, 1021, 856, 646, 540, 
                                      439, 353, 247, 171, 119, 77, 72, 56, 41, 
                                      26, 20, 10, 10, 10, 5, 3, 0, 1, 1, 3, 0, 
                                      1, 1))
ggplot(size_dist, aes(x = transaction_size, y = frequency)) +
  geom_bar(stat="identity")

# transaction size distribution
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 0.000   2.000   3.000   4.383   6.000  30.000 

# 9835 transactions, 125 unique items sold, 0.03506172*(9835*125) = 43104 
# items sold. Average number of items purchased per transaction = 43104/9835 = 
# 4.383. 2163 transactions contained 1 item only. A median of 3 items/transaction 
# tells us half of the transactions were of less than 3 items, while half were 
# of more than 3 items.


#inspect(transactions) # takes some time to print
size(transactions) # most transactions were of less than 10 items
LIST(transactions) # rows
itemLabels(transactions) # columns
itemFrequency(transactions)

# top ten most-purchased items. Of the top 10, 5 were desktops, 3 were laptops, 
# 1 earpod, and 1 monitor
itemFrequencyPlot(transactions, type = "relative", topN = 10)
itemFrequencyPlot(transactions, type = "absolute", topN = 10) 

# items with minimum support = 0.1
itemFrequencyPlot(transactions, support = 0.1) 

image(transactions[150:300])
image(sample(transactions, 150)) # displays the sparse matrix with a subset of
# randomly chosen transactions, since we can't fit whole data set on screen.
# No noticable variation in items bought throughout the month as a function
# of time (assuming the data is in chronological order). Some items where 
# purchased more often than others.


#####################################
# Find and Evaluate Association Rules
#####################################

rules <- apriori(transactions, parameter = list(support = 0.1, confidence = 0.8, 
                                                minlen = 2))
# returns rules that meet the specified minimum support, minimum confidence, and 
# minimum number of total items per rule. 

# The above yielded zero rules. To tuning!

# Support tuning: 
# Default min support of 0.1 is too high. An item set must appear in 0.1*9835 ~ 
# 984 transactions to be considered in making rules. We will lower the support 
# so there's more itemsets to make rules with. On the other hand a min support 
# that is too low doesn't really cut down on the search space.

# Confidence tuning: 
# We would like confidence high to give us strong rules. However, if too high
# then the only rules we'll get will be obvious ones, and obvious ones might
# not generate more revenue since people are buying them together anyways.
# We would like to see unobvious insights into the large data set. If too low 
# however, we might get overwhelmed with spurious rules.
 
rules <- apriori(transactions, parameter = list(support = 60/9835, 
                                                confidence = 0.25, minlen = 2))
summary(rules)
inspect(rules[is.redundant(rules)]) # shows the redundant rules. a rule X -> Y 
# is redundant if for some subset X' of X, conf(X'->Y) >= conf(X->Y)

rules_pruned <- rules[!is.redundant(rules)] # delete redundant rules, which do
# not offer additional insight

summary(rules_pruned) # 586 rules found. Rule sizes were no larger than 4. At 
# least 75% of the rules were of size 2 or 3. Many of the rules' supports and 
# confidences were higher than the minimum, which indicates our minimum support
# and confidence was not too high.
write(rules_pruned, "rules.csv", sep =",") 

inspect(sort(rules_pruned, by = "support")[1:10]) # top ten most frequent
inspect(sort(rules_pruned, by = "confidence")[1:10]) # top ten confidence
inspect(sort(rules_pruned, by = "lift")[1:20]) # top ten most important 
inspect(subset(rules_pruned, lift > 3.5))

# all rules containing a specific item
HPblacktricolorink_rules <- subset(rules_pruned, 
                                  items %in% "HP Black & Tri-color Ink") 
inspect(HPblacktricolorink_rules)
summary(HPblacktricolorink_rules)
write(HPblacktricolorink_rules, "hpblackandtricolorinkrules.csv", sep =",")

# all rules containing a specific item
applemagickeyboard_rules <- subset(rules_pruned, 
                                   items %in% "Apple Magic Keyboard") 
inspect(applemagickeyboard_rules)
summary(applemagickeyboard_rules)
write(applemagickeyboard_rules, "applemagickeyboardrules.csv", sep =",")

# all rules containing a specific item
dell2desktop_rules <- subset(rules_pruned, items %in% "Dell 2 Desktop") 
inspect(dell2desktop_rules)
summary(dell2desktop_rules)
write(dell2desktop_rules, "dell2desktoprules.csv", sep =",")

# all rules containing a specific item
imac_rules <- subset(rules_pruned, items %in% "iMac") 
inspect(imac_rules)
summary(imac_rules)
write(imac_rules, "imacrules.csv", sep =",")


# any rule with iMac on RHS.
# shows what items led to increased sales of iMacs
rhs_rules <- subset(rules_pruned, rhs %in% "iMac") 
inspect(rhs_rules)
summary(rhs_rules)

# rules with either iMac OR Apple Wireless Keyboard on the RHS. %in% = logical
# or
rhs_rules2 <- subset(rules_pruned, rhs %in% c("iMac","Apple Wireless Keyboard"))
inspect(rhs_rules2)
summary(rhs_rules2)

# rules with both iMac AND Apple Wireless Keyboard on the RHS. %ain% = logical
# and
rhs_rules_3 <- subset(rules_pruned, rhs %ain% c("iMac","Apple Wireless Keyboard"))
inspect(rhs_rules_3)
summary(rhs_rules_3)

# partial matching - all rules containing "Desktop" in the names of items
rhs_rules_4 <- subset(rules_pruned, items %pin% "Desktop") 
inspect(rhs_rules_4)
summary(rhs_rules_4)

inspect(subset(rules_pruned, lhs %ain% c("iMac","Apple Wireless Keyboard")))

plot(rules_pruned, "scatterplot") # shades of red represent lift. Most 
# important rules have low support, and varied confidence.
plot(rules_pruned, "two-key plot") # colors represent rule length. Shorter 
# length rules were more common.

plot(rules_pruned[1:100], "graph", control=list(type="iMac"))
plot(HPblacktricolorink_rules, "graph")
inspect(HPblacktricolorink_rules)
plot(dell2desktop_rules, "graph")
inspect(dell2desktop_rules)

# Stop cluster when you are done
stopCluster(cl) 