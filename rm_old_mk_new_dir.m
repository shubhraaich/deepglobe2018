function rm_old_mk_new_dir(dir_)

if isdir(dir_)
    assert(rmdir(dir_,'s'));
end
assert(mkdir(dir_));

end