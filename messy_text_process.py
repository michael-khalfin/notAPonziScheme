in_file = open("covariance_matrix.txt", "r")

out_text = ""

lines = in_file.readlines()
for line in lines:
  out_text = out_text + line.replace(" ", ",")

in_file.close()


out_file = open("new_covariance_matrix.csv", "w")
out_file.write(out_text)
out_file.close()
