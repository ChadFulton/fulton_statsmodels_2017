import subprocess

scripts = [
    'c2_inf.py', 'c2_nile.py', 'c2_rbc.py',
    'c3_inf.py', 'c3_nile.py', 'c3_point.py',  # 'c3_rbc.py'
    'c4_inf.py', 'c4_nile.py', 'c4_rbc.py',
    'c5_inf.py', 'c5_nile.py', 'c5_rbc.py',
    'c6_inf.py', 'c6_nile.py', 'c6_rbc.py',
]

if __name__ == '__main__':
    with open('run_all_stdout.txt', 'w') as outfile:
        with open('run_all_stderr.txt', 'w') as errfile:

            for script in scripts:
                print('- Running %s' % script)
                outfile.write('- %s %s' % (script, '-' * 50))
                errfile.write('- %s %s' % (script, '-' * 50))
                cmd = 'python %s' % script
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                stdout, stderr = p.communicate()

                outfile.write(stdout)
                errfile.write(stderr)