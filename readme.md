目前dataloader:
加载event对应的txt + EST变成对应的event volume

目前模型结构：
在main model上加一层

处理event的流程：在原来的处理流程上+process_event_t_bin+copy_bin_from_filtered