import java.io.BufferedWriter;
import java.io.FileWriter;

public class Bitmaps
{
   public void createfile()
   {

      try{
            DibDump dib = new DibDump();
            String outputfile = "targetInOut.txt";
            BufferedWriter out = new BufferedWriter(new FileWriter(outputfile));
            out.write("5");
            out.newLine();
            dib.makePels("hands/hand_1.bmp", outputfile);
            dib.makePels("hands/hand_2.bmp", outputfile);
            dib.makePels("hands/hand_3.bmp", outputfile);
            dib.makePels("hands/hand_4.bmp", outputfile);
            dib.makePels("hands/hand_5.bmp", outputfile);
            out.close();
      }
      catch(Exception e){
            System.out.println(e);
      }
      try{
         String outputfile = "targetInOut.txt";
         BufferedWriter out = new BufferedWriter(new FileWriter(outputfile, true));
         out.write("0 0.25 0.5 0.75 1.0");
         out.close();
      }
      catch(Exception e){
         System.out.println(e);
      }

   }
}