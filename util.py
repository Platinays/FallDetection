__author__ = 'Shuo Yu'


dict_sensor_id = {
    5: 'E8BD107D58B4',
    4: 'EE9F6185DA6C',
    2: 'EB0BE26E8C52',
    3: 'F7FCFFD2F166',
    1: 'D202B31CD2C3',
}

class DatabaseInterface:
    def __init__(self, db_name=None):
        self.cur = self.connect()
        self.ret_list = []
        self.db_name = db_name

    def connect(self):
        """
        Connect to the database
        :return:
        """
        import pymysql

        return pymysql.connect(host="127.0.0.1",
                           user="shuoyu",
                           passwd="qoowpyep",
                           db="silverlink",
                           charset='utf8',
                           autocommit=True).cursor()

    def clear(self):
        """
        Data read are accumulated! Remember to clear them when attempting to read new data!
        :return:
        """
        self.ret_list = []

    def set_db_name(self, db_name):
        self.db_name = db_name

    def read_from_db(self, **kwargs):
        """
        Get sensor data from param 'db_name', and save to the buffer 'ret_list'.
        If not specified, self.db_name will be used.
        Specify 'sensor_id', 'subject_id', 'label_id'.
        'get_ts' can be specified to include timestamps. Disabled by default.

        :param kwargs:
        :return:
        """
        self.clear()
        sql = '''
            SELECT timestamp, x_accel, y_accel, z_accel
            FROM %s
            WHERE sensor_id = '%s' AND subject_id = %d AND label_id = %d
            ORDER BY timestamp
        ''' % (kwargs['db_name'] if 'db_name' in kwargs else self.db_name,
               dict_sensor_id[kwargs['sensor_id']], kwargs['subject_id'], kwargs['label_id'])


        self.cur.execute(sql)   # execute sql command

        if 'get_ts' in kwargs and kwargs['get_ts'] == True:
            for row in self.cur:
                self.ret_list.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
        else:
            for row in self.cur:
                self.ret_list.append([int(row[1]), int(row[2]), int(row[3])])

    def read_from_db_by_ts(self, **kwargs):
        """
        Get sensor data from 'db_name', and save to the buffer 'ret_list'.
        If not specified, self.db_name will be used.
        Specify 'subject_id', 'ts_start', 'ts_end'.
        'find_falls' can be specified to enable fall-only fetching (by checking 'label_id' != 0)
        'get_ts' can be specified to include timestamps. Disabled by default.

        :param kwargs:
        :return:
        """
        self.clear()
        sql = '''
            SELECT timestamp, x_accel, y_accel, z_accel
            FROM %s
            WHERE subject_id = '%s' AND timestamp >= '%s' AND timestamp <= '%s' ''' \
              % (kwargs['db_name'], kwargs['subject_id'], kwargs['ts_start'], kwargs['ts_end']) \
              if 'db_name' in kwargs else (self.db_name, kwargs['subject_id'], kwargs['ts_start'], kwargs['ts_end'])

        if kwargs['find_falls'] == True:
            sql += ' AND label_id != 0'

        self.cur.execute(sql)   # execute sql command
        if 'get_ts' in kwargs and kwargs['get_ts'] == True:
            for row in self.cur:
                self.ret_list.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
        else:
            for row in self.cur:
                self.ret_list.append([int(row[1]), int(row[2]), int(row[3])])

    def _write_to_db(self, ins):
        """
        Write the param 'ins' into the db.
        Modify this method to meet the actual need.
        :param insert:
        :return:
        """
        sql = '''
            INSERT INTO %s (sensor_id, subject_id, label_id, freq, timestamp, x_accel, y_accel, z_accel)
            VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
        '''
        if ins[1] == 'null':
            return

        timestamp = int(ins[0])
        if ins[1].endswith('b'):
            subject_id = int(ins[1][:-1]) + 100
        else:
            subject_id = int(ins[1])
        label_id = int(ins[2])
        sensor_id = ins[3]
        freq = ins[4]
        x_accel = ins[5]
        y_accel = ins[6]
        z_accel = ins[7]
        while True:
            try:
                self.cur.execute(sql % (self.db_name, sensor_id, subject_id, label_id, freq, timestamp, x_accel, y_accel, z_accel))
                break
            except Exception as e:
                print(e)
                print('Retrying by increasing the timestamp...')
                timestamp += 1

    def convert_csv_to_db_by_pattern(self, pattern):
        import glob
        for file in glob.glob(pattern):
            with open(file, 'r') as fh:
                print('Current file: %s' % file)
                for line in fh:
                    temp = line.split(',')
                    self._write_to_db(temp)

    def convert_csv_to_db_by_file(self, file):
        with open(file, 'r') as fh:
            print('Current file: %s' % file)
            for line in fh:
                temp = line.split(',')
                self._write_to_db(temp)

    def convert_matlab_to_db_by_pattern(self, pattern):
        import h5py, glob, numpy as np
        sql = '''
            INSERT INTO %s (subject_id, label_id, timestamp, x_accel, y_accel, z_accel)
            VALUES ('%s', '%s', '%s', '%s', '%s', '%s')
        '''
        # label_id refers to is_fall in the mat file

        subject_id = 0
        for file in glob.glob(pattern):
            subject_id += 1
            print('%s: %s' % (subject_id, file))
            d = h5py.File(file)
            rows = np.matrix(d['tmp']).T[:, [0, 2, 3, 4, -1]].tolist()
            for row in rows:
                label_id = row[-1]
                x_accel = round(float(row[1]) * 100)
                y_accel = round(float(row[2]) * 100)
                z_accel = round(float(row[3]) * 100)
                timestamp = round(row[0] * 1000)
                try:
                    self.cur.execute(sql % (self.db_name, subject_id, label_id, timestamp, x_accel, y_accel, z_accel))
                except Exception as e:
                    print(e)

    def output_as_csv(self, file):
        """
        Output self.ret_list as csv file
        :param file:
        :return:
        """
        write_to_csv(self.ret_list, file)


def write_to_csv(accel_list, file):
    with open(file, 'w') as csv:
        for row in accel_list:
            csv.write(('{:f},' * len(row)).format(*row))
            csv.write('\n')


if __name__ == '__main__':
    di = DatabaseInterface(db_name='test_data_160801')
    di.convert_csv_to_db_by_pattern(r'C:\Users\shuoyu\Google Drive\SilverLink\test_clinic\160801\*.csv')
