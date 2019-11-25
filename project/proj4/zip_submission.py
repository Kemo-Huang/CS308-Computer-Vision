import os
import shutil

def copy_files(src, dest):
  try:
    if not os.path.exists(dest):
      os.makedirs(dest)
    for f in os.listdir(src):
      if os.path.isfile(os.path.join(src, f)):
        shutil.copyfile(os.path.join(src, f), os.path.join(dest,f))
  except shutil.Error as e:
    print('Directory not copied. Error: %s' % e)
  except OSError as e:
    print('Directory not copied. Error: %s' % e)

shutil.rmtree('temp_submission', ignore_errors=True)
os.mkdir('temp_submission')
for dir_name in ['code']:
  copy_files(dir_name, '/'.join(['temp_submission', dir_name]))
shutil.make_archive('submission', 'zip', 'temp_submission')
shutil.rmtree('temp_submission', ignore_errors=True)