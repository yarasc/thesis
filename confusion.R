library(ggplot2)
library("ggpubr")
require(gridExtra)
require(dplyr)
require(tidyr)
library(data.table)
library(ConfusionTableR)
library("cvms")
library("ggnewscale")
library(rsvg)


data_path <- "/Users/zeehondje/Documents/GitHub/thesis/"
path <- "test10DVS2RnnMseB04_1"


file <- paste(data_path,path,".csv", sep = "")

df <- read.csv(file)

df$Prediction<-sub(".","",as.character(df$Prediction))
df$Prediction<-gsub(".{1}$","",as.character(df$Prediction))
df$Target<-sub(".","",as.character(df$Target))
df$Target<-gsub(".{1}$","",as.character(df$Target))

dt <- data.table(df)

dt <- separate(dt, Prediction, into=c("P1","P2","P3","P4","P5","P6","P7","P8"), sep = ", ")
dt <- separate(dt, Target, into=c("T1","T2","T3","T4","T5","T6","T7","T8"), sep = " ")
dt <- unite(dt, P1,T1, col = B1, sep =",")
dt <- unite(dt, P2,T2, col = B2, sep =",")
dt <- unite(dt, P3,T3, col = B3, sep =",")
dt <- unite(dt, P4,T4, col = B4, sep =",")
dt <- unite(dt, P5,T5, col = B5, sep =",")
dt <- unite(dt, P6,T6, col = B6, sep =",")
dt <- unite(dt, P7,T7, col = B7, sep =",")
dt <- unite(dt, P8,T8, col = B8, sep =",")
dt <- pivot_longer(dt, cols = c("B1","B2","B3","B4","B5","B6","B7","B8"),names_to = "batch", values_to = "Prec" )
dt <- separate(dt, Prec, into=c("Prediction", "Target"), sep = ",")

dx<-dt[dt$Epoch==9,]
conf_mat <- confusion_matrix(targets = dx$Target,
                             predictions = dx$Prediction)
plot_confusion_matrix(conf_mat$`Confusion Matrix`[[1]],
                      add_normalized = FALSE, 
                      add_row_percentages = FALSE,
                      add_col_percentages = FALSE,
                      add_sums = TRUE,
                      add_arrows = TRUE, 
                      add_counts = TRUE,
                      add_zero_shading = FALSE,
                      )

      