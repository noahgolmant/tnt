from torch.autograd import Variable

class Engine(object):
    def __init__(self, create_graph=False, mini_batch_size=-1):
        self.hooks = {}
        self.create_graph = create_graph
        self.mini_batch_size = mini_batch_size

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, iterator, maxepoch, optimizer):
        state = {
            'network': network,
            'iterator': iterator,
            'maxepoch': maxepoch,
            'optimizer': optimizer,
            'epoch': 0,
            't': 0,
            'train': True,
        }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                inputs = Variable(cast(sample[0], 'float'))
                targets = Variable(cast(sample[1], 'long'))

                batch_size = len(inputs)

                # Chunk into equally sized mini-batches for batch size statistics
                mini_inputs = inputs.chunk(batch_size // self.mini_batch_size)
                mini_targets = targets.chunk(batch_size // self.mini_batch_size)
                mini_samples = zip(mini_inputs, mini_targets)

                state['optimizer'].zero_grad()
                for k, mini_sample in enumerate(mini_samples):
                    state['sample'] = mini_sample
                    self.hook('on_sample', state)

                    def closure():
                        loss, output = state['network'](state['sample'])
                        state['output'] = output
                        state['loss'] = loss
                        loss.backward(create_graph=self.create_graph, retain_graph=self.create_graph)
                        self.hook('on_forward', state)
                        # to free memory in save_for_backward
                        state['output'] = None
                        state['loss'] = None
                        return loss

                state['optimizer'].step(closure, scaling_factor=len(mini_inputs), grad_clip=5.)
                self.hook('on_update', state)

                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def test(self, network, iterator):
        state = {
            'network': network,
            'iterator': iterator,
            't': 0,
            'train': False,
        }

        self.hook('on_start', state)
        for sample in state['iterator']:
            state['sample'] = sample
            self.hook('on_sample', state)

            def closure():
                loss, output = state['network'](state['sample'])
                state['output'] = output
                state['loss'] = loss
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            closure()
            state['t'] += 1
        self.hook('on_end', state)
        return state
