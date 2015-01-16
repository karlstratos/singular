Singular
========
#### Under construction

To truncate lower dimensions and retain, say, top 100 PCA dimensions, use the
following command:
`awk '{DIM=100; for(i=1;i<=2+DIM;++i) {printf $i " ";} printf "\n";}' [original] > [truncated]`
