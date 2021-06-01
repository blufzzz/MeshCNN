from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, pr, re = model.test()
        writer.update_counter(ncorrect, nexamples)
        writer.update_counter_pr_re(pr, re)
    writer.print_acc(epoch, writer.acc)
    print(writer.acc, writer.opr, writer.ore)
    return writer.acc, writer.opr, writer.ore

if __name__ == '__main__':
    run_test()
