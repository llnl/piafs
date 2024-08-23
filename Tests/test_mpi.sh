#!/bin/bash

root_dir=$PWD

if [ -z "$PIAFS2D_DIR" ]; then
  echo "Environment variable PIAFS2D_DIR not set."
  echo "Please set it as the path to PIAFS2D source."
  exit
fi

# PIAFS2D location (PIAFS2D_DIR should exist in the environment)
piafs2d_dir=$PIAFS2D_DIR
# other PIAFS2D-related stuff
piafs2d_exec="PIAFS2D"

#export env vars for other scripts to run PIAFS2D
export PIAFS2D_EXEC_W_PATH="${piafs2d_dir}/bin/${piafs2d_exec}"
export MPI_EXEC="mpiexec --oversubscribe"
export PIAFS2D_EXEC_OTHER_ARGS=""

# some details about PIAFS2D benchmarks
# (benchmark solutions maintained on the public repository)
# do not change these, unless you know what you are doing
piafs2d_benchmarks_repo="ssh://git@czgitlab.llnl.gov:7999/piafs/piafs2d_benchmarks.git"
piafs2d_benchmarks_branch="master"
piafs2d_benchmarks_dir="benchmarks"

# stuff about test directory
piafs2d_test_dir="_test"
exclude_flag="--exclude={'op*','initial*','out.log','README.md','.git*'}"
diff_filelistname="diff_file_list"
diff_file="diff.log"

# other stuff
RUN_SCRIPT="run.sh"
DISABLED=".disabled"

if [ -f "$PIAFS2D_EXEC_W_PATH" ]; then

  echo "PIAFS2D binary found."
  echo "-------------------------"

else

  echo "---------------------------------"
  echo "ERROR !!!"
  echo " "
  echo "PIAFS2D binary NOT FOUND !!!"
  echo " "
  echo "$PIAFS2D_EXEC_W_PATH does not exist"
  echo " "
  echo "---------------------------------"

fi

# compile the PIAFS2D diff code
piafs2d_diff_srcname="Extras/piafs2dDiff_RegTests.c"
PIAFS2D_DIFF="PIAFS2D_DIFF"
if [ -f "$piafs2d_dir/$piafs2d_diff_srcname" ]; then
  echo "Compiling PIAFS2D-diff."
  gcc $piafs2d_dir/$piafs2d_diff_srcname -lm -o $root_dir/$PIAFS2D_DIFF
else
  echo "---------------------------------"
  echo "ERROR !!!"
  echo " "
  echo "PIAFS2D-Diff source NOT FOUND !!!"
  echo " "
  echo "$piafs2d_dir/$piafs2d_diff_srcname does not exist"
  echo " "
  echo "---------------------------------"
fi
PIAFS2D_DIFF_CMD="$root_dir/$PIAFS2D_DIFF -r 1.0e-14 "

# clone benchmarks
if [ -d "$piafs2d_benchmarks_dir" ]; then
  cd $piafs2d_benchmarks_dir
  if [ -d ".git" ]; then
    echo "benchmarks directory already exists. refreshing it."
    git reset HEAD --hard
    git checkout $piafs2d_benchmarks_branch
    git pull --force
    cd ../
  else
    echo "benchmarks directory exists, but not a git repo. erasing..."
    cd $root_dir
    rm -rf $piafs2d_benchmarks_dir
    echo "getting PIAFS2D benchmarks from $piafs2d_benchmarks_repo ($piafs2d_benchmarks_branch branch)"
    git clone $piafs2d_benchmarks_repo $piafs2d_benchmarks_dir
    cd $piafs2d_benchmarks_dir
    git checkout $piafs2d_benchmarks_branch
  fi
else
  echo "getting PIAFS2D benchmarks from $piafs2d_benchmarks_repo ($piafs2d_benchmarks_branch branch)"
  git clone $piafs2d_benchmarks_repo $piafs2d_benchmarks_dir
  cd $piafs2d_benchmarks_dir
  git checkout $piafs2d_benchmarks_branch
fi
cd $root_dir
echo "-------------------------"

# create test dir and copy input files
timestamp=`date | sed -e 's/ /_/g' -e 's/:/./g'`
test_dirname=${piafs2d_test_dir}_${timestamp}
rm -rf $test_dirname && mkdir $test_dirname
echo "copying test cases to $test_dirname ..."
rsync_cmd="rsync -a $exclude_flag $root_dir/$piafs2d_benchmarks_dir/ $root_dir/$test_dirname/"
eval $rsync_cmd

# run the cases
cd $root_dir/$test_dirname

echo "PIAFS2D Tests"
echo "Date/Time       : $(date '+%d/%m/%Y %H:%M:%S')"
echo "PIAFS2D repo      : $piafs2d_repo"
echo "PIAFS2D branch    : $piafs2d_branch"
echo "benchmarks repo  : $piafs2d_benchmarks_repo"
echo "benchmarks branch: $piafs2d_benchmarks_branch"
echo " "
echo " "

n_pass=0
n_fail=0
n_skip=0
echo "running tests..."
echo " "
for f in *; do
  if [ -d "$f" ]; then
    echo "entering $f..."
    cd $f
    if [ -f "$DISABLED" ]; then
      echo "Skipping; $f test is disabled."
      ((n_skip+=1))
    else
      if [ -f "$RUN_SCRIPT" ]; then
        chmod +x $RUN_SCRIPT && ./$RUN_SCRIPT
        while read F  ; do
          echo "    comparing $F ..."
          result=$($PIAFS2D_DIFF_CMD $F $root_dir/$piafs2d_benchmarks_dir/$f/$F 2>&1 >> $diff_file)
          if [ -z "$result" ]; then
            if [ -s "$diff_file" ]; then
              ((n_fail+=1))
              echo "                        **DIFFERENCES FOUND**"
              echo "### Dump of PIAFS2D-diff output ####"
              cat $diff_file
              echo "### End ####"
              echo " "
            else
              ((n_pass+=1))
              echo "                        **passed**"
            fi
          else
            ((n_fail+=1))
            echo "                        **FILE COMPARISON FAILED**"
            echo "Command output: $result "
            echo " "
            echo "### Dump of screen output   ####"
            cat out.log
            echo "### End ####"
            echo "### Directory contents   ####"
            ls -lh ./
            echo "### End ####"
            echo "### Benchmark directory contents   ####"
            ls -lh $root_dir/$piafs2d_benchmarks_dir/$f/
            echo "### End ####"
          fi
        done <./$diff_filelistname
      else
        echo "Error: $RUN_SCRIPT not found."
      fi
    fi
    echo " "
    cd ../
  fi
done
echo "done."
echo "-------------------------"
cd $root_dir

echo "all done. Bye!"
echo "$n_pass file comparisons **passed**"
if [[ $n_fail -gt 0 ]]; then
  echo "$n_fail file comparisons **failed**"
  exit 1
fi
echo "-------------------------"

rm -rf $root_dir/$PIAFS2D_DIFF
