#!/bin/bash

root_dir=$PWD

if [ -z "$PIAFS_DIR" ]; then
  echo "Environment variable PIAFS_DIR not set."
  echo "Please set it as the path to PIAFS source."
  exit
fi

# PIAFS location (PIAFS_DIR should exist in the environment)
piafs_dir=$PIAFS_DIR
# other PIAFS-related stuff

# Executable selection:
# - If PIAFS_EXEC_W_PATH is already set, use it.
# - Else if PIAFS_EXEC is set, use ${PIAFS_DIR}/bin/${PIAFS_EXEC}.
# - Else try ${PIAFS_DIR}/bin/PIAFS (legacy name).
# - Else auto-detect the newest executable matching ${PIAFS_DIR}/bin/PIAFS*
if [ -z "$PIAFS_EXEC_W_PATH" ]; then
  if [ -n "$PIAFS_EXEC" ]; then
    PIAFS_EXEC_W_PATH="${piafs_dir}/bin/${PIAFS_EXEC}"
  else
    if [ -x "${piafs_dir}/bin/PIAFS" ]; then
      PIAFS_EXEC_W_PATH="${piafs_dir}/bin/PIAFS"
    else
      # Prefer MPI builds if present, otherwise take newest PIAFS* in bin/
      cand=$(ls -1t "${piafs_dir}/bin"/PIAFS* 2>/dev/null | head -n 1)
      cand_mpi=$(ls -1t "${piafs_dir}/bin"/PIAFS*-mpi* 2>/dev/null | head -n 1)
      if [ -n "$cand_mpi" ]; then
        PIAFS_EXEC_W_PATH="$cand_mpi"
      else
        PIAFS_EXEC_W_PATH="$cand"
      fi
    fi
  fi
fi

#export env vars for other scripts to run PIAFS
export PIAFS_EXEC_W_PATH
export PIAFS_EXEC_OTHER_ARGS=""

# Set MPI executor - use MPIEXEC if provided, otherwise use mpiexec with --oversubscribe
if [ -z "$MPIEXEC" ]; then
  export MPI_EXEC="mpiexec --oversubscribe"
  # Test if --oversubscribe flag is supported
  $MPI_EXEC -n 1 echo "test" &> /dev/null
  if [ $? -ne 0 ]; then
    echo "Warning: --oversubscribe flag not supported, falling back to 'mpiexec'"
    export MPI_EXEC="mpiexec"
  fi
else
  export MPI_EXEC="$MPIEXEC"
  echo "Using custom MPI executor: $MPI_EXEC"
fi


# Root directory of the test execution, usually the build directory
test_run_dir=$PIAFS_DIR/test_run_temp

# Clean and create the test_run_dir
rm -rf $test_run_dir && mkdir -p $test_run_dir

# Benchmarks will be checked out here
benchmarks_checkout_dir=$test_run_dir/benchmarks

# do not change these, unless you know what you are doing
piafs_benchmarks_repo="ssh://git@czgitlab.llnl.gov:7999/piafs/piafs_benchmarks.git"
piafs_benchmarks_branch="master"
piafs_benchmarks_dir=$test_run_dir/benchmarks


# stuff about test directory
piafs_test_dir="_test"
exclude_flag="--exclude={'op*','initial*','out.log','README.md','.git*'}"
diff_filelistname="diff_file_list"
diff_file="diff.log"

# other stuff
RUN_SCRIPT="run.sh"
DISABLED=".disabled"

if [ -x "$PIAFS_EXEC_W_PATH" ]; then

  echo "PIAFS binary found."
  echo "-------------------------"
  echo "Using: $PIAFS_EXEC_W_PATH"

else

  echo "---------------------------------"
  echo "ERROR !!!"
  echo " "
  echo "PIAFS binary NOT FOUND !!!"
  echo " "
  echo "$PIAFS_EXEC_W_PATH does not exist or is not executable"
  echo "Looked in: ${piafs_dir}/bin"
  echo " "
  echo "---------------------------------"

fi

# compile the PIAFS diff code
piafs_diff_srcname="Extras/piafsDiff_RegTests.c"
PIAFS_DIFF="PIAFS_DIFF"

# Use PIAFS_SRC_DIR if set (for out-of-source builds), otherwise use PIAFS_DIR
if [ -z "$PIAFS_SRC_DIR" ]; then
  piafs_src_dir=$piafs_dir
else
  piafs_src_dir=$PIAFS_SRC_DIR
fi

if [ -f "$piafs_src_dir/$piafs_diff_srcname" ]; then
  echo "Compiling PIAFS-diff."
  gcc $piafs_src_dir/$piafs_diff_srcname -lm -o $test_run_dir/$PIAFS_DIFF
else
  echo "---------------------------------"
  echo "ERROR !!!"
  echo " "
  echo "PIAFS-Diff source NOT FOUND !!!"
  echo " "
  echo "$piafs_src_dir/$piafs_diff_srcname does not exist"
  echo " "
  echo "---------------------------------"
fi
PIAFS_DIFF_CMD="$test_run_dir/$PIAFS_DIFF -r 1.0e-14 "

# clone benchmarks
if [ -d "$piafs_benchmarks_dir" ]; then
  cd $piafs_benchmarks_dir
  if [ -d ".git" ]; then
    echo "benchmarks directory already exists. refreshing it."
    git reset HEAD --hard
    git checkout $piafs_benchmarks_branch
    git pull --force
    cd ../
  else
    echo "benchmarks directory exists, but not a git repo. erasing..."
    cd $root_dir
    rm -rf $piafs_benchmarks_dir
    echo "getting PIAFS benchmarks from $piafs_benchmarks_repo ($piafs_benchmarks_branch branch)"
    git clone $piafs_benchmarks_repo $piafs_benchmarks_dir
    cd $piafs_benchmarks_dir
    git checkout $piafs_benchmarks_branch
  fi
else
  echo "getting PIAFS benchmarks from $piafs_benchmarks_repo ($piafs_benchmarks_branch branch)"
  git clone $piafs_benchmarks_repo $piafs_benchmarks_dir
  cd $piafs_benchmarks_dir
  git checkout $piafs_benchmarks_branch
fi
cd $root_dir
echo "-------------------------"

# create test dir and copy input files
timestamp=`date | sed -e 's/ /_/g' -e 's/:/./g'`
test_dirname=${piafs_test_dir}_${timestamp}
rm -rf $test_run_dir/$test_dirname && mkdir $test_run_dir/$test_dirname
echo "copying test cases to $test_dirname ..."
rsync_cmd="rsync -a $exclude_flag $piafs_benchmarks_dir/ $test_run_dir/$test_dirname/"
eval $rsync_cmd

# run the cases
cd $test_run_dir/$test_dirname

echo "PIAFS Tests"
echo "Date/Time       : $(date '+%d/%m/%Y %H:%M:%S')"
echo "PIAFS repo      : $piafs_repo"
echo "PIAFS branch    : $piafs_branch"
echo "benchmarks repo  : $piafs_benchmarks_repo"
echo "benchmarks branch: $piafs_benchmarks_branch"
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
          result=$($PIAFS_DIFF_CMD $F $piafs_benchmarks_dir/$f/$F 2>&1 >> $diff_file)
          if [ -z "$result" ]; then
            if [ -s "$diff_file" ]; then
              ((n_fail+=1))
              echo "                        **DIFFERENCES FOUND**"
              echo "### Dump of PIAFS-diff output ####"
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
            ls -lh $root_dir/$piafs_benchmarks_dir/$f/
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

rm -rf $test_run_dir/$PIAFS_DIFF
