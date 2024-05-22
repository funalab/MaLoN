exp <- data.frame()
for( i in 1:4 ){
  exp_i <- t( read.table( paste( "./Exp", i, ".txt", sep="" ), sep="\t", row.names=1 ) )[ ,-1 ]
  exp <- rbind( exp, exp_i )
}
write.table( exp, "./ecoli_sospathway_concatenated.tsv", sep="\t", quote=F, row.names=F )
